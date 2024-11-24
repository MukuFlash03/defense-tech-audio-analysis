def format_translated_conversation(chat_completion_response):
    content = chat_completion_response.choices[0].message.content
  
    lines = content.split('\n')
    
    formatted_lines = [line.strip() for line in lines if line.strip()]
    
    formatted_output = '\n'.join(formatted_lines)
    
    return formatted_output
