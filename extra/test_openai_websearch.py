from openai import OpenAI
from constants import api_key

client = OpenAI(api_key=api_key)


completion = client.chat.completions.create(
    model="gpt-4o-mini-search-preview",
    web_search_options={"search_context_size": "medium"},
    messages=[
        {
            "role": "system",
            "content": (
                "نام چند خبرگزاری به تو داده می شود. بگو که هریک وابسته به چه نهادی و با چه طیف عقیدتی و حزبی ای می باشد؟ به ترتیب به صورت json پاسخ بده."
                "مثلا یک خبرگزاری ممکن است وابسته به سپاه باشد و یکی وابسته به حزب اصلاح طلب."
                )
        },
        {
            "role": "user",
            "content": (
                "خبرگزاری فارس\nخبرگزاری تسنیم\nخبرگزاری ایسنا"
            )
        }
    ],
    # temperature=0.2,
    max_tokens=4000
)

print(completion.choices[0].message.content)
print()

# response = client.responses.create(
#     model="gpt-4o-mini-search-preview",  # or "o4-nano" depending on the deployment environment
#     # tools=[{
#     #     "type": "web_search_preview",
#     #     "search_context_size": "low"  # can be "low", "medium", or "high"
#     # }],
#     input=[
#         {
#             "role": "system",
#             "content": (
#                 "You are an expert fact-checker. You must verify all information "
#                 "using current web results before answering. If a claim is true, say 'True'; "
#                 "if false, say 'False'; if unclear, say 'Uncertain'. Provide a brief explanation "
#                 "and cite any web sources used."
#             )
#         },
#         {
#             "role": "user",
#             "content": "What movie won Best Picture in 2025?"
#         }
#     ]
# )
#
# # Print the result
# print(response.output_text)
