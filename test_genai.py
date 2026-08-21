import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from google import genai

client = genai.Client(api_key="AQ.Ab8RN6KyQ11Sp7lbEwAQ1wV_rbC2tT-OT0s7cqVnm_oJmb-Qlw")

prompt = """আপনি একজন বাংলা ভাষাবিদ ও সাইবার হেট স্পিচ বিশ্লেষক। 
প্রদত্ত বাংলা কমেন্ট এবং তার লেবেল দেখে কেন এই মন্তব্যটি এই লেবেল পেয়েছে তার একটি ২-৩ বাক্যের বাংলা ব্যাখ্যা লিখুন।

কমেন্ট: "এই হিন্দুগুলোকে দেশ থেকে তাড়িয়ে দেওয়া উচিত"
ঘৃণার ধরন: Religious Hate
লক্ষ্য: Community
তীব্রতা: Severe

ব্যাখ্যা:"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("--- GENERATED BENGALI EXPLANATION ---")
print(response.text.strip())
