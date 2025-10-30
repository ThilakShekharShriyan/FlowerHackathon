from google import genai
c = genai.Client()
print('HAS_RESPONSES', hasattr(c,'responses'))
print('HAS_MODELS', hasattr(c,'models'))
print('CLIENT_ATTRS', [a for a in dir(c) if not a.startswith('_')])
print('MODELS_ATTRS', [a for a in dir(c.models) if not a.startswith('_')])
print('CHATS_ATTRS', [a for a in dir(c.chats) if not a.startswith('_')])
