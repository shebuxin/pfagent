from openai import OpenAI


client = OpenAI(api_key="")
# client = OpenAI()


# models = client.models.list()
# for model in models.data:
#     print(model.id)


# models = client.models.list()
# fine_tuned_models = [model for model in models.data if model.id.startswith("ft-")]
# for ft_model in fine_tuned_models:
#     print(ft_model.id)


models = client.models.list()

# Filter to include only fine-tuned models that are not checkpoint step models.
final_models = [
    model for model in models.data
    if model.id.startswith("ft:") and "ckpt-step" not in model.id
]

for model in final_models:
    print(model.id)
