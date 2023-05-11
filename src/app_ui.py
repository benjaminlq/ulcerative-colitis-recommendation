welcome_msg = """This is an AI physician assistant that at advises on time to next colonoscopy based on guidelines from the USA. Please enter a short description of your patient case, including the age, possible risk factors, and prior colonoscopy results if available. For example:
\n1/ A 45 year old Chinese woman with no family history of colon cancer and otherwise well. There is no previous colonoscopy
\n2/ A 65 year old man with colonoscopy this year and 10 3-5mm hyperplastic polyps in the descending colon that were removed
"""
need_api_key_msg = "Welcome! This app is a physician assistant that provides the interval to the next colonoscopy by analysing patient risk factors and prior colonoscopy findings. It is powered by OpenAI's text models: gpt-3.5-turbo and gpt-4 (if you have access to it). To get started, simply enter your OpenAI API Key below."
helper_api_key_prompt = "The model comparison tool works best with pay-as-you-go API keys. For more information on OpenAI API rate limits, check [this link](https://platform.openai.com/docs/guides/rate-limits/overview).\n\n- Don't have an API key? No worries! Create one [here](https://platform.openai.com/account/api-keys).\n- Want to upgrade your free-trial API key? Just enter your billing information [here](https://platform.openai.com/account/billing/overview)."
helper_api_key_placeholder = "Paste your OpenAI API key here (sk-...)"

