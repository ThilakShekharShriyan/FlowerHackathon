from google import genai
import inspect
c = genai.Client()
fn = c.models.generate_content
print('SIG', inspect.signature(fn))
