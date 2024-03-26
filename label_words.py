import requests
from bs4 import BeautifulSoup
from transformers import RobertaTokenizer

def get_related_words(word):
    url = f"https://relatedwords.org/relatedto/{word}"
    response = requests.get(url)
    print(response.status_code)
    soup = BeautifulSoup(response.content, 'html.parser')
    words = soup.find_all('div', class_='words')[0].find_all('li')
    related_words = []
    for word in words:
        related_words.append(word.text)
    return related_words


candidate_words_clash_sarc = ['not','kidding', 'irony', 'joking', 'sarcastic', 'jesting', 'teasing', 'joke', 'ironic', 'sarcasm', 'ridiculous', 'playing']
candidate_words_clash_nonsarc = ['true', 'serious', 'real']
candidate_words_ques_sarc = ['yes']
candidate_words_ques_nonsarc = ['no']

def return_label_words(prompt_type = 'clash'):
    if prompt_type == 'clash':
        return candidate_words_clash_nonsarc, candidate_words_clash_sarc
    else:
        return candidate_words_ques_nonsarc, candidate_words_ques_sarc

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
candidate_words = candidate_words_clash_sarc
candidate_words.extend(candidate_words_clash_nonsarc)
added_words = []
for word in candidate_words:
    if word in tokenizer.get_vocab():
        continue
    else:
        added_words.append(word)

added_length = tokenizer.add_tokens(added_words)

#added_length = len(added_words)

print(f'the added words are{added_words} \n and the length is {added_length}')
