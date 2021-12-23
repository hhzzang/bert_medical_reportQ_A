import os, json


root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dev_input_path = os.path.join(root_path,'data','dev', 'dev_input.txt"')
dev_answer_path = os.path.join(root_path,'data','dev', 'dev_answer.txt')


with open(dev_input_path, 'r', encoding='utf8') as f1, open(dev_answer_path, encoding='utf8') as f2:
    for i, j in zip(f1, f2):
        temp = i.split('\t')
        context = temp[0].strip()
        question = temp[1].strip()
        text = j.strip()
        idx = context.find(text)
        temp = {'paragraphs': [
            {'qas': [{'answers': [{'text': text, 'answer_start': idx}], 'id': None, 'question': question}],
             'context': context}]}
        data.append(temp)


res = {'version':'process_valid.json','data':data}


with open('dev.json', 'w', encoding='utf-8') as make_file:
    json.dump(res, make_file)


