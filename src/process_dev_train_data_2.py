import json,re,os
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
train_path = os.path.join(root_path,'data','train', 'MedQuAD_v1.0_train.json')
dev_path = os.path.join(root_path,'src','dev.json')

train_data = json.load(open(train_path))
dev_data = json.load(open(dev_path))


def sen_sep(test):
    test = test.split('.')
    if len(test) > 1:
        temp1 = test.pop()
        temp2 = test.pop()
        res = test + [".".join([temp2, temp1])]
    else:
        res = test
    return res


def sliding_window(test_context, text_answer, test_question, max_len=500):
    ###인풋예시###
    # max_len = 505 # 최대 길이 값
    # test_context = train_data['data'][276]['paragraphs'][0]['context']
    # text_answer = train_data['data'][276]['paragraphs'][0]['qas'][0]['answers'][0]['text']
    # test_question = train_data['data'][276]['paragraphs'][0]['qas'][0]['question']

    ### context 본문나누기####
    temp_precont = sen_sep(test_context[0])
    temp_sufcont = sen_sep(test_context[1])

    ## answer나누기
    temp_res = sen_sep(text_answer)

    split_master = []
    # 처음에 하나 추가
    temp_dict = {}
    permit = max_len - len(test_question)
    for j in range(len(temp_sufcont)):
        permit -= len(temp_sufcont[j])
        if permit < 0:
            inc_last_idx = j
            break
    first_context = ".".join(temp_sufcont[:j])
    temp_dict['head'] = ''
    temp_dict['tail'] = first_context
    temp_dict['question'] = test_question
    temp_dict['head_answer'] = ''
    temp_dict['tail_answer'] = text_answer
    split_master.append(temp_dict)

    # 기준선에 따라 나누기
    for i in range(len(temp_res)):
        temp_dict = {}
        head = i + 1

        head_answer = ".".join(temp_res[:head])  # 위에 포함 할 정답 값

        start_idx = None  # 기준선이다 헷갈리지 말자

        # 전반부 context포함 answer
        permit = max_len - len(head_answer) - len(test_question)

        for q in range(len(temp_precont)):
            inc_idx = len(temp_precont) - q - 1  # 인덱스 0부터 시작해서 하나 뺀다
            permit -= len(temp_precont[inc_idx])
            if permit < 0:
                inc_idx = inc_idx + 1
                break

        head_context = ".".join(temp_precont[inc_idx:]) + head_answer

        # 후반부 context포함 answer
        tail_answer = ".".join(temp_res[head:])  # 밑에 포함할 정답 값
        permit = max_len - len(test_question)

        for j in range(len(temp_sufcont[head:])):
            inc_last_idx = head + j
            permit -= len(temp_sufcont[inc_last_idx])
            # 설마 문장 하나의 길이가 512자를 안 넘겟지? 믿고 쓰는 조건
            if permit < 0:
                inc_last_idx = inc_last_idx - 1
                break

        if len(temp_sufcont[head:]) == 0:
            inc_last_idx = head

        if head == len(temp_res):  # 맨마지막 tail은 정답이 들어가 있지 않은 window라 제거
            tail_context = ''
        else:
            tail_context = ".".join(temp_sufcont[head:inc_last_idx + 1])

        temp_dict['head'] = head_context
        temp_dict['tail'] = tail_context
        temp_dict['question'] = test_question
        temp_dict['head_answer'] = head_answer
        temp_dict['tail_answer'] = tail_answer
        split_master.append(temp_dict)

    return split_master


# [{},{},{}..]
# train_data['data'][m]['paragraphs'][0]['context'] m번째 콘텍스트
# train_data['data'][m]['paragraphs'][0]['qas'][ans]['answers'][0]['text'] m번째 콘텍스트 중에 ans번째 answer
# test_question = train_data['data'][m]['paragraphs'][0]['qas'][ans]['question']
# max_len = 500

##### dev_process_json데이터 만들기
total_data = []
for m in range(len(dev_data['data'])):
    for ans in range(len(dev_data['data'][m]['paragraphs'][0]['qas'])):
        temp_data = {}
        answer_start = dev_data['data'][m]['paragraphs'][0]['qas'][ans]['answers'][0]['answer_start']
        v1 = dev_data['data'][m]['paragraphs'][0]['context']
        v1 = v1[:answer_start] + '*_*' + v1[answer_start:]
        v1 = v1.split('*_*')

        v2 = dev_data['data'][m]['paragraphs'][0]['qas'][ans]['answers'][0]['text']
        v3 = dev_data['data'][m]['paragraphs'][0]['qas'][ans]['question']

        key = 'key'
        temp_data[key] = sliding_window(v1, v2, v3)
        total_data.append(temp_data)


make_data = []
for i in range(len(total_data)):
    for j in range(len(total_data[i]['key'])):
        head_context = total_data[i]['key'][j]['head']  # context
        tail_context = total_data[i]['key'][j]['tail']
        question = text = total_data[i]['key'][j]['question']
        if head_context != '':
            head_text = total_data[i]['key'][j]['head_answer']
            head_idx = head_context.find(head_text)
            data = {'paragraphs': [{'qas': [
                {'answers': [{'text': head_text, 'answer_start': head_idx, 'id': 'head'}], 'question': question}],
                                    'context': head_context}]}
            make_data.append(data)
        if tail_context != '':
            tail_text = total_data[i]['key'][j]['tail_answer']
            tail_idx = tail_context.find(tail_text)
            data = {'paragraphs': [{'qas': [
                {'answers': [{'text': tail_text, 'answer_start': tail_idx, 'id': 'tail'}], 'question': question}],
                                    'context': tail_context}]}
            make_data.append(data)

dict_result = {}
dict_result['version'] = 'made_validset'
dict_result['data'] = make_data

with open('process_val.json', 'w', encoding='utf-8') as make_file:
    json.dump(dict_result, make_file)


##### train_process_json데이터 만들기

total_data = []
for m in range(len(train_data['data'])):
    for ans in range(len(train_data['data'][m]['paragraphs'][0]['qas'])):
        temp_data = {}
        answer_start = train_data['data'][m]['paragraphs'][0]['qas'][ans]['answers'][0]['answer_start']
        v1 = train_data['data'][m]['paragraphs'][0]['context']
        v1 = v1[:answer_start] + '*_*' + v1[answer_start:]
        v1 = v1.split('*_*')

        v2 = train_data['data'][m]['paragraphs'][0]['qas'][ans]['answers'][0]['text']
        v3 = train_data['data'][m]['paragraphs'][0]['qas'][ans]['question']

        key = 'key'
        temp_data[key] = sliding_window(v1, v2, v3)
        total_data.append(temp_data)

##### dev_process_json데이터 만들기
make_data = []
for i in range(len(total_data)):
    for j in range(len(total_data[i]['key'])):
        head_context = total_data[i]['key'][j]['head']  # context
        tail_context = total_data[i]['key'][j]['tail']
        question = text = total_data[i]['key'][j]['question']
        if head_context != '':
            head_text = total_data[i]['key'][j]['head_answer']
            head_idx = head_context.find(head_text)
            data = {'paragraphs': [{'qas': [
                {'answers': [{'text': head_text, 'answer_start': head_idx, 'id': 'head'}], 'question': question}],
                                    'context': head_context}]}
            make_data.append(data)
        if tail_context != '':
            tail_text = total_data[i]['key'][j]['tail_answer']
            tail_idx = tail_context.find(tail_text)
            data = {'paragraphs': [{'qas': [
                {'answers': [{'text': tail_text, 'answer_start': tail_idx, 'id': 'tail'}], 'question': question}],
                                    'context': tail_context}]}
            make_data.append(data)

dict_result = {}
dict_result['version'] = 'made_trainset'
dict_result['data'] = make_data

with open('process_MedQuAD_v1.0_train.json', 'w', encoding='utf-8') as make_file:
    json.dump(dict_result, make_file)