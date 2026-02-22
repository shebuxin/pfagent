from openai import OpenAI

client = OpenAI(api_key="")
# client = OpenAI()


file = client.files.create(
  file=open("fine_tuning_data.jsonl", "rb"),
  purpose="fine-tune"
)

client.fine_tuning.jobs.create(
  training_file=file.id,
  model="gpt-4o-mini-2024-07-18"
)


# train = client.files.create(
#     file=open("train.jsonl", "rb"),
#     purpose="fine-tune"
# )

# validation = client.files.create(
#     file=open("validation.jsonl", "rb"),
#     purpose="fine-tune"
# )

# job = client.fine_tuning.jobs.create(
#     training_file=train.id,
#     validation_file=validation.id,
#     model="gpt-4o-mini-2024-07-18"
# )

# print(job)
# print(job.fine_tuned_model)
