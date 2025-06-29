from models.chat import chatbot_respond
from utils.translator import translate_to_english

def chatbot_reply(user_input):
    text = translate_to_english(user_input)
    response = chatbot_respond(text)
    return response