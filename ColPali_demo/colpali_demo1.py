import sys, os, time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image
from io import BytesIO

from pdf2image import convert_from_path
from pypdf import PdfReader

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_utils import scale_image, get_base64_image

def get_pdf_images(pdf_filename):
    reader = PdfReader(pdf_filename)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    images = convert_from_path(pdf_filename)
    assert len(images) == len(page_texts)
    return (images, page_texts)

print("PT1")

if torch.cuda.is_available():
  device = torch.device("cuda")
  type = torch.bfloat16
# elif torch.backends.mps.is_available():
  # device = torch.device("mps")
  # type = torch.float32
else:
  device = torch.device("cpu")
  type = torch.float32
print("DEVICE",device)

"""Example script to run inference with ColPali"""
# Load model
model_name = "vidore/colpali"
model = ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=type).eval()
model.load_adapter(model_name)
model.to(device)
processor = AutoProcessor.from_pretrained(model_name)

sample_pdfs = [ { "filename": fn}  for fn in os.listdir('.') if fn.endswith('.pdf') ]

print("PT2")

# retrieve images
all_doc_images = []
for pdf in sample_pdfs:
    (page_images,page_texts) = get_pdf_images(pdf["filename"])
    pdf["images"] = page_images
    pdf["texts"]  = page_texts
    all_doc_images.extend(page_images)

print("PT3")

# run inference - docs
img_dataloader = DataLoader(
    all_doc_images,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: process_images(processor, x),
)

ds = []
for batch_doc in tqdm(img_dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

print("PT4")

# queries = ["From which university does James V. Fiorca come ?"]
# queries = ["Composition of the LoTTE benchmark"]
queries = ["What is the approximate number of Australian men that studied engineering in 1992?"]

# run inference - queries
query_dataloader = DataLoader(
    queries,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: process_queries(processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
)

qs = []
for batch_query in query_dataloader:
    with torch.no_grad():
        batch_query = {k: v.to(model.device) for (k,v) in batch_query.items()}
        embeddings_query = model(**batch_query)
    qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

print("PT5")

# run evaluation
retriever_evaluator = CustomEvaluator(is_multi_vector=True)
scores = retriever_evaluator.evaluate(qs, ds)
print(scores)
print("-"*80)
print(scores.argmax(axis=1))
all_doc_images[scores.argmax(axis=1)[0]].save('temp.png', 'PNG')

print("PT6")

