import logging
import os
from flask import Flask, render_template, request, jsonify, session
from ollama import AsyncClient
import hypercorn
from hypercorn.config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "chatbot_secret_key"

# Initialize Ollama client
ollama_client = AsyncClient()

# Define tool functions
def add_two_numbers(a: int, b: int) -> int:
    return a + b

def subtract_two_numbers(a: int, b: int) -> int:
    return a - b

# Available tools mapping
available_functions = {
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
}

# Tool definitions
add_numbers_tool = add_two_numbers
subtract_numbers_tool = {
    'type': 'function',
    'function': {
        'name': 'subtract_two_numbers',
        'description': 'Subtract two numbers',
        'parameters': {
            'type': 'object',
            'required': ['a', 'b'],
            'properties': {
                'a': {'type': 'integer', 'description': 'The first number'},
                'b': {'type': 'integer', 'description': 'The second number'},
            },
        },
    },
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = request.json
        message = data.get('message', '')
        mode = data.get('mode', 'chat')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Initialize messages list for the session if it doesn't exist
        if 'messages' not in session:
            session['messages'] = []

        # Add user message to history
        session['messages'].append({
            'role': 'user',
            'content': message
        })

        try:
            if mode == 'generate':
                try:
                    response = await ollama_client.generate(
                        'llama2',
                        prompt=message,
                        stream=False
                    )
                    bot_response = response['response']
                except Exception as e:
                    logger.error(f"Generate mode error: {str(e)}")
                    raise
            elif mode == 'structured':
                # Handle structured output requests
                try:
                    # Detect the type of structured output needed
                    if 'friends' in message.lower():
                        schema = FriendList.model_json_schema()
                        response = await ollama_client.chat(
                            model='llama2',
                            messages=session['messages'],
                            format=schema,
                            options={'temperature': 0}
                        )
                        # Validate response with Pydantic
                        structured_response = FriendList.model_validate_json(response['message']['content'])
                        bot_response = structured_response.model_dump_json()
                    elif 'weather' in message.lower():
                        schema = WeatherInfo.model_json_schema()
                        response = await ollama_client.chat(
                            model='llama2',
                            messages=session['messages'],
                            format=schema,
                            options={'temperature': 0}
                        )
                        structured_response = WeatherInfo.model_validate_json(response['message']['content'])
                        bot_response = structured_response.model_dump_json()
                    else:
                        # Default to regular chat if no structure detected
                        response = await ollama_client.chat(
                            model='llama2',
                            messages=session['messages']
                        )
                        bot_response = response['message']['content']
                except Exception as e:
                    logger.error(f"Structured output error: {str(e)}")
                    raise
            else:
                from flask import Response, stream_with_context
                
                async def generate_stream():
                    response = await ollama_client.chat(
                        model='llama2',
                        messages=session['messages'],
                        tools=[add_numbers_tool, subtract_numbers_tool],
                        stream=True
                    )
                    
                    full_response = ""
                    async for chunk in response:
                        if chunk.message.tool_calls:
                            # Handle tool calls
                            for tool in chunk.message.tool_calls:
                                if function_to_call := available_functions.get(tool.function.name):
                                    output = function_to_call(**tool.function.arguments)
                                    session['messages'].append({
                                        'role': 'tool',
                                        'content': str(output),
                                        'name': tool.function.name
                                    })
                            continue
                            
                        content = chunk.message.content
                        full_response += content
                        yield f"data: {content}\n\n"
                    
                    # Save complete response to session
                    session['messages'].append({
                        'role': 'assistant',
                        'content': full_response
                    })
                    
                return Response(
                    stream_with_context(generate_stream()),
                    mimetype='text/event-stream'
                )

            # Add bot response to history
            session['messages'].append({
                'role': 'assistant',
                'content': bot_response
            })

            return jsonify({
                'response': bot_response,
                'history': session['messages']
            })

        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            return jsonify({'error': 'Model response failed'}), 500

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({'error': 'Failed to process request'}), 500

@app.route('/clear', methods=['POST'])
async def clear_history():
    session['messages'] = []
    return jsonify({'status': 'success'})

@app.route('/messages', methods=['GET'])
async def get_messages():
    try:
        history = session.get('history', [])
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching messages: {str(e)}")
        return jsonify([])

@app.route('/history', methods=['GET'])
async def get_history():
    try:
        history = session.get('history', [])
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify([])

@app.route('/history/<int:history_id>', methods=['DELETE'])
async def delete_history(history_id):
    try:
        history = session.get('history', [])
        if 0 <= history_id < len(history):
            del history[history_id]
            session['history'] = history
            return jsonify({'status': 'success'})
        return jsonify({'error': 'History not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history/<int:history_id>', methods=['PUT'])
async def edit_history(history_id):
    try:
        data = request.json
        history = session.get('history', [])
        if 0 <= history_id < len(history):
            history[history_id]['content'] = data['content']
            session['history'] = history
            return jsonify({'status': 'success'})
        return jsonify({'error': 'History not found'}), 404
    except Exception as e:
        logger.error(f"Error editing history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html'), 500

if __name__ == '__main__':
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    hypercorn.run(app, config)