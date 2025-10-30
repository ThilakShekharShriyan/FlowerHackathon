from google import genai
c = genai.Client()
name = next(n.name for n in c.models.list() if "gemini-2.0-flash" in n.name)
print('USING', name)
resp = c.models.generate_content(model=name, contents="Say Ready plainly.")
print('TEXT', getattr(resp,'output_text',None) or getattr(resp,'text',None))
