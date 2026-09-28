import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

text = "AI: [Amit, laughing] एक मजाक सुनाओ, जो तुम्हें पसंद आए: \n\nएक आदमी ने डॉक्टर से कहा, 'मुझे पता नहीं है कि मैं क्या बीमार हूँ, लेकिन मुझे एक बुरा स्वाद है।' 😊"
c1 = re.sub(r'\[.*?\]', '', text)
c2 = re.sub(r'[\U00010000-\U0010ffff]', '', c1)
c2 = re.sub(r'[\u2600-\u27bf]', '', c2).strip()
# Remove AI: prefix
c2 = re.sub(r'^AI:\s*', '', c2).strip()
print("Original:", text)
print("Cleaned:", c2)
