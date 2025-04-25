# 📊 Analisador de Sentimentos com spaCy (via Regras)

Este projeto utiliza a biblioteca [spaCy](https://spacy.io/) para **detectar o sentimento de frases** com base em **regras simples de palavras-chave**, usando os componentes `Matcher` e `PhraseMatcher`.

## 🚀 Objetivo

Classificar frases como **POSITIVO**, **NEGATIVO** ou **NEUTRO** com base em palavras e expressões detectadas no texto. Isso é feito **sem treinamento de modelo**, apenas por meio de padrões linguísticos definidos manualmente.

---

## 🧠 Como funciona

O sistema segue os seguintes passos:

1. **Cria uma pipeline vazia do spaCy para português**.
2. **Define listas de palavras e expressões positivas e negativas**.
3. **Cria dois tipos de matchers:**
   - `Matcher`: para palavras individuais.
   - `PhraseMatcher`: para detectar expressões compostas.
4. **Cria uma extensão personalizada no objeto `Doc` do spaCy**, que calcula o sentimento com base nas correspondências.
5. **Analisa uma lista de frases e imprime o sentimento de cada uma**.

---

## 📦 Dependências

- Python 3.7+
- spaCy (v3+)

Instalação:
```bash
pip install spacy
python -m spacy download pt_core_news_sm  # Caso queira usar um modelo treinado futuramente
```

---

## 🧾 Código completo comentado

```python
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc

# 1. Criar pipeline spaCy em branco para português
nlp = spacy.blank("pt")

# 2. Listas de palavras e expressões
palavras_positivas = ["incrível", "amei", "divertido", "adorei", "maravilhosa", "obra-prima", "bom", "superou", "gostei", "linda", "recomendo"]
palavras_negativas = ["péssimo", "cansativo", "sem graça", "ruim", "chato", "decepcionante", "horrível", "sono", "mal feito", "sem sentido"]

# 3. Criar os matchers
matcher = Matcher(nlp.vocab)  # Para palavras simples
phrase_matcher = PhraseMatcher(nlp.vocab)  # Para expressões compostas

# Adicionar padrões de palavras (MATCHER)
for palavra in palavras_positivas:
    matcher.add("POSITIVO", [[{"LOWER": palavra}]])
for palavra in palavras_negativas:
    matcher.add("NEGATIVO", [[{"LOWER": palavra}]])

# Adicionar padrões de frases (PHRASEMATCHER)
positivas_doc = list(nlp.pipe(palavras_positivas))  # Transforma lista de strings em Docs
negativas_doc = list(nlp.pipe(palavras_negativas))
phrase_matcher.add("POSITIVO", None, *positivas_doc)  # None = usar ORTH como atributo
phrase_matcher.add("NEGATIVO", None, *negativas_doc)

# 4. Criar extensão personalizada no objeto Doc
def detectar_sentimento(doc):
    matches = matcher(doc) + phrase_matcher(doc)
    positivos = sum(1 for match_id, _, _ in matches if nlp.vocab.strings[match_id] == "POSITIVO")
    negativos = sum(1 for match_id, _, _ in matches if nlp.vocab.strings[match_id] == "NEGATIVO")
    
    if positivos > negativos:
        return "POSITIVO"
    elif negativos > positivos:
        return "NEGATIVO"
    elif positivos == negativos and positivos > 0:
        return "NEUTRO"
    else:
        return "NEUTRO"

# Adiciona o atributo personalizado ._.sentimento
Doc.set_extension("sentimento", getter=detectar_sentimento)

# 5. Frases de teste
frases = [
    "Esse filme foi incrível! Amei cada parte.",
    "O filme foi péssimo, perdi meu tempo.",
    "Achei o filme ok, nada demais.",
    "Muito divertido, ri bastante!",
    "Cansativo e sem graça."
]

# 6. Executar a análise
print("🔍 Análise baseada em regras:\n")
for frase in frases:
    doc = nlp(frase)
    print(f"{frase} → {doc._.sentimento}")
```

---

## 📈 Resultado esperado

```
🔍 Análise baseada em regras:

Esse filme foi incrível! Amei cada parte. → POSITIVO  
O filme foi péssimo, perdi meu tempo. → NEGATIVO  
Achei o filme ok, nada demais. → NEUTRO  
Muito divertido, ri bastante! → POSITIVO  
Cansativo e sem graça. → NEGATIVO  
```

---

## 🧩 Possibilidades de expansão

- Adicionar pesos para diferentes palavras (por exemplo, "obra-prima" pode ter peso maior).
- Combinar com um modelo `textcat` treinado para um sistema híbrido.
- Aplicar pré-processamento (remoção de stopwords, lematização).
- Transformar isso em uma API com Flask ou FastAPI.
