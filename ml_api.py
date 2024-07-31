import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from decoder import Decoder
from encoder import Encoder
from seq2seq import Seq2Seq
import pickle


class TranslateRequest(BaseModel):
    oreacion_en: str

app = FastAPI()

# Variables
oracion_en: str 
attention:str
device = torch.device('cpu') 
model_path = "C:\\Users\\augus\\OneDrive\\Documentos Compartidos\\Facultad\\Redes Neuronales\\Final\\tut6-model.pt"

with open("vocab_src", 'rb') as f:
      vocab_src = torch.load(f, device)

with open("vocab_tgt", 'rb') as f:
      vocab_tgt = torch.load(f, device)

with open("itos_vocab_src", 'rb') as f:
     itos_vocab_src = torch.load(f, device)

with open("itos_vocab_tgt", 'rb') as f: 
      itos_vocab_tgt = torch.load(f, device)


INPUT_DIM = len(itos_vocab_src)
OUTPUT_DIM = len(itos_vocab_tgt)
HID_DIM = 256 
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8 
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

SRC_PAD_IDX = vocab_src['<pad>']
TGT_PAD_IDX = vocab_tgt['<pad>']


def translate_sentence(sentence, vocab_src, itos_vocab_src, vocab_tgt, itos_vocab_tgt, model, device, max_len = 50):

    model.eval()

    tokens = [token.lower() for token in sentence]

    tokens = tokens[1:]

    src_indexes = [vocab_src[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [vocab_tgt['<bos>']]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == vocab_tgt['<eos>']:
            break

    #trg_tokens = [vocab_tgt.get_itos() for i in trg_indexes]
    trg_tokens = [itos_vocab_tgt[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def agregar_tokens(sentence):
    return f"<bos> {sentence} <eos>"


model = Seq2Seq(enc, dec, SRC_PAD_IDX, TGT_PAD_IDX, device).to(device)
model.load_state_dict(torch.load(model_path, device))


@app.post("/model/translate")
def traslate(translate_request: TranslateRequest): 
    oracion_es = ""
    print(translate_request.oreacion_en)
    sentence_mod = agregar_tokens(translate_request.oreacion_en)
    print(sentence_mod)
    sentence_t = sentence_mod.split()
    print(sentence_t)
    oracion_es, attention = translate_sentence(sentence_t, vocab_src, itos_vocab_src, vocab_tgt, itos_vocab_tgt, model, device)
    print(oracion_es)
    return JSONResponse(content={"traduccion": oracion_es})


