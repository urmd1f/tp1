def set_hidden(info):
    # hidden_cnt : 하나의 은닉계층인 경우 해당 은닉계층이 갖는 노드의 수
	# hidden_config : 다수의 은닉계층인 경우 해당 은닉계층들의 수와 노드의 수
    global hidden_cnt, hidden_config

    # 은닉계층 셋팅 과정에서 int 형식으로 들어오는 경우와 리스트 형식으로 들어오는 경우를 분리하였습니다. 
	# int 형식은 하나의 은닉계층으로 고정
    # list 형식은 하나 이상의 은닉계층 
    # isinstance() 는 첫 번째 매개변수 타입과 두 번째 매개변수 타입이 일치하는지 확인합니다. 
    if isinstance(info, int):
        hidden_cnt = info      # int 형식이 맞는 경우 hidden_cnt 변수에 은닉 계층의 노드 수를 할당 
        hidden_config = None   # int 형식이 맞는 경우 hidden_config 변수는 None 처리
    
	# 사용자의 설정값이 list 형식인 경우 해당 info 에 담긴 값은 hidden_config 변수에 할당
    else:
        hidden_config = info

# 기존에 파라미터를 초기화 하는 메서드의 재정의 입니다. 
def init_param():
    
	# 사용자가 정의한 은닉 계층의 정보에 따라 hidden_config 변수의 타입은 None 혹은 List로 지정됩니다. 
	# hidden_config 변수의 타입이 None 이 아닌 경우 하나 이상의 은닉 계층을 갖으므로, 그에 따른 파라미터 초기화 메서드를 동작시킵니다. 
    if hidden_config is not None:
		# 사용자를 위한 은닉계층 안내 문구 출력 및 다수의 파라미터 생성을 위한 메서드 동작 
        print(f"[안내] 은닉 계층 {len(hidden_config)}개를 갖는 다층 퍼셉트론이 적용됩니다.")
        init_param_hiddens() # 본 메서드는 곧 정의할 예정입니다. 
	
    # hidden_config 변수의 타입이 None 인 경우 하나의 은닉 계층을 갖으므로, 그에 따른 파라미터 초기화 메서드를 동작시킵니다. 
    else:
		# 사용자를 위한 은닉계층 안내 문구 출력 및 다수의 파라미터 생성을 위한 메서드 동작 
        print("[안내] 은닉 계층 하나를 갖는 다층 퍼셉트론이 적용됩니다.")
        init_param_hidden1() # 본 메서드는 곧 정의할 예정입니다.


# # [테스트 코드]
# # 테스트를 위해 간단히 메서드의 동작 확인만 수행할 수 있도록 정의하였습니다. 
# def init_param_hiddens():
#     print("init_param_hiddens() 가 동작하였습니다.")

# def init_param_hidden1():
#     print("init_param_hidden1() 가 동작하였습니다.")

# # 하나 이상의 은닉계층을 설정하는 경우
# set_hidden([5,3])
# init_param()

# # 하나의 은닉계층만을 설정하는 경우
# set_hidden(5)
# init_param()

# 기존에 신경망의 선형연산을 수행하는 메서드를 재정의 합니다. 
# 선형연산에 필요한 독립변수가 담긴 변수를 매개변수로 취합니다. 
def forward_neuralnet(x):
    # 사용자가 정의한 은닉 계층에 따라 hidden_config 변수의 타입은 None 혹은 List로 지정됩니다. 
    # hidden_config 변수의 타입이 None 이 아닌 경우 하나 이상의 은닉 계층을 갖으므로, 그에 따른 선형연산 메서드를 동작시킵니다. 
    if hidden_config is not None:
        return forward_neuralnet_hiddens(x)
    
    # hidden_config 변수의 타입이 None 인 경우 하나의 은닉 계층을 갖으므로, 그에 따른 선형연산 메서드를 동작시킵니다. 
    else:
        return forward_neuralnet_hidden1(x)

# 기존에 경사하강법에 따른 파라미터 업데이트 메서드를 재정의 합니다. 
# 위 방식과 마찬가지로 hidden_config 변수타입에 맞춰 동작하는 메서드를 달리합니다. 
# 매개변수는 경사하강법에 필요한 변수들을 취합니다. 
def backprop_neuralnet(G_output, hiddens):
    if hidden_config is not None:
        backprop_neuralnet_hiddens(G_output, hiddens)
    else:
        backprop_neuralnet_hidden1(G_output, hiddens)

# # [테스트 코드]
# # 메서드 동작 테스트에 필요한 내부 메서드는 사전에 간단히 정의합니다.
# def forward_neuralnet_hidden1(x):
#     print("forward_neuralnet_hidden1() 메서드가 동작하였습니다.")

# def forward_neuralnet_hiddens(x):
#     print("forward_neuralnet_hiddens() 메서드가 동작하였습니다.")

# def backprop_neuralnet_hidden1(G_output, hiddens):
#     print("backprop_neuralnet_hidden1() 메서드가 동작하였습니다.")

# def backprop_neuralnet_hiddens(G_output, hiddens):
#     print("backprop_neuralnet_hiddens() 메서드가 동작하였습니다.")


# # 은닉 계층이 하나 이상인 경우에 대한 동작 테스트
# set_hidden([5,3])
# forward_neuralnet(x=[1,2,3])
# backprop_neuralnet(G_output=0, hiddens=[0])

# # 은닉 계층이 하나인 경우에 대한 동작 테스트
# set_hidden(5)
# forward_neuralnet(x=[1,2,3])
# backprop_neuralnet(G_output=0, hiddens=[0])

# 계층 사이의 파라미터 초기화를 진행하는 메서드
# shape : 다음과 같은 리스트 형식으로 값을 전달받습니다. 
# ex) 독립변수 5개, 은닉계층의 노드 3개인 경우 [5,3]
# ex) 은닉계층의 노드 수 3개, 출력계층의 노드 수 1개인 경우 [3,1]
def allocate_param_pair(shape):
    # [가중치 초기화 과정]
    weight = np.random.normal(RND_MEAN, RND_STD, shape)
		
	# [편향 초기화 과정]
    # 편향은 리스트 형식으로 값을 전달받는 것을 고려하여 -1 인덱스를 활용 
    bias   = np.zeros(shape[-1])
    
    # 파이썬의 데이터 타입 중 딕셔너리 구조로 정의합니다.
	# 가중치는 'w', 편향은 'b' 키로 접근하여 활용할 수 있도록 합니다. 
    return {'w':weight, 'b':bias}

# # [테스트 코드]
# import numpy as np
# RND_MEAN, RND_STD = 0.0, 1.0

# # 입력 벡터의 노드 수 5개, 은닉 계층의 노드 수 3개 인 경우 해당 계층 사이에 존재하는 파라미터는 다음과 같습니다. 
# # 가중치 15개, 편향 3개
# pm_hidden = allocate_param_pair([5,3])

# print("pm_hidden['w'] : \n", pm_hidden['w'])
# print("pm_hidden['b'] : \n", pm_hidden['b'])

# 하나의 은닉계층을 갖는 신경망의 파라미터변수 생성 메서드
def init_param_hidden1():

    # 추후 활용을 위한 전역변수 지정 
    global pm_output, pm_hidden, input_cnt, output_cnt, hidden_cnt
    # input_cnt  : 사전 신경망 모델이 들어있는 파일에 정의된 입력 계층의 노드 수 (독립변수의 수)
	# output_cnt : 사전 신경망 모델이 들어있는 파일에 정의된 출력 계층의 노드 수 (종속변수의 수) 
    # pm_hidden : 입력계층과 은닉계층 사이에 존재하는 파라미터 
    # pm_output : 은닉계층과 출력계층 사이에 존재하는 파라미터
	
	# 사전에 정의한 allocate_param_pair() 메서드를 통해 계층과 계층사이의 파라미터가 담긴 변수를 생성합니다.
    pm_hidden = allocate_param_pair([input_cnt, hidden_cnt])
    pm_output = allocate_param_pair([hidden_cnt, output_cnt])

# # [테스트 코드]
# # 다층 퍼셉트론 신경망 구조 설정
# input_cnt = 10  # 입력계층의 노드 수 10개 
# set_hidden(5)   # 은닉계층의 노드 수 5개
# output_cnt = 1  # 출력계층의 노드 수 1개 

# # 파라미터 초기화 메서드 동작 시나리오
# # init_param() -> init_param_hidden1() -> allocate_param_pair()
# init_param()

# print("pm_hidden['w'] : \n", pm_hidden['w'])
# print("pm_hidden['b'] : \n", pm_hidden['b'])
# print("pm_output['w'] : \n", pm_output['w'])
# print("pm_output['b'] : \n", pm_output['b'])

# 다수의 은닉계층이 있는 경우 각 계층 사이에 존재하는 파라미터 변수 생성 메서드
def init_param_hiddens():
    # input_cnt    : 입력계층의 노드 수
    # output_cnt   : 출력계층의 노드 수 
    # hidden_config : 은닉계층에 대한 정보 
    # pm_hiddens: [입력계층]-> (파라미터{pm_hiddens}) ->[은닉계층]--> (파라미터{pm_hiddens}) -->[마지막 은닉계층]--> ...
    # pm_output : [마지막 은닉계층]-> (파라미터{pm_output}) ->[출력계층]
    global pm_output, pm_hiddens, input_cnt, output_cnt, hidden_config

    # 입력계층부터 마지막 은닉계층 사이에 존재하는 파라미터 값을 저장하기 위한 빈리스트 사전 정의 
    # 마지막 은닉계층과 출력 계층 사이의 파라미터는 따로 변수를 둘 예정 
    pm_hiddens = []

    # 반복문을 활용하여 파라미터를 생성할 예정이기에 input_cnt 변수의 값을
    # prev_cnt 변수에 할당합니다. 
    prev_cnt = input_cnt

    # hidden_config:list -> 은닉계층의 수와 폭 정보
    # Ex) 6, [3,7], [3,6,9], ...
    for hidden_cnt in hidden_config:
        
        # 입력계층(prev_cnt)과 첫번째 은닉계층(hidden_cnt)의 파라미터 생성
        # 해당 계층 사이의 파라미터는 pm_hiddens 변수에 쌓는다. 
        pm_hiddens.append(allocate_param_pair([prev_cnt, hidden_cnt]))
        
        # 은닉 계층의 노드 수 정보를 prev_cnt 변수로 할당하여 allocate_param_pair()를 통해
        # 다시 은닉 계층들 사이의 파라미터 변수를 생성할 수 있도록 합니다. 
        prev_cnt = hidden_cnt
        # 모든 은닉계층들의 파라미터 생성을 마치면 반복문 탈출

    # 마지막 은닉계층(prev_cnt)과 출력계층(output_cnt) 사이의 파라미터 생성 및 변수 생성  
    pm_output = allocate_param_pair([prev_cnt, output_cnt])

# # [테스트 코드]
# # 다층 퍼셉트론 신경망 구조 설정
# input_cnt = 10    # 입력계층의 노드 수 10개 
# set_hidden([5,3]) # 은닉계층들의 노드 수 각 5개, 3개
# output_cnt = 1    # 출력계층의 노드 수 1개 

# # 파라미터 초기화 메서드 동작 시나리오
# # init_param() -> init_param_hiddens() -> allocate_param_pair()
# init_param()

# print("입력계층과 첫 번째 은닉계층 사이의 가중치 : \n", pm_hiddens[0]['w'])
# print("입력계층과 첫 번째 은닉계층 사이의 편향 : \n", pm_hiddens[0]['b'])
# print("첫 번째 은닉계층과 두 번째 은닉계층 사이의 가중치 : \n", pm_hiddens[1]['w'])
# print("첫 번째 은닉계층과 두 번째 은닉계층 사이의 편향 : \n", pm_hiddens[1]['b'])
# print("두 번째 은닉계층과 출력계층 사이의 가중치 : \n", pm_output['w'])
# print("두 번째 은닉계층과 출력계층 사이의 편향 : \n", pm_output['b'])

# 활성화 함수 ReLU() 정의 
def relu(x):
    # 음수와 0을 걸러주는 numpy 내부 메서드 활용 
    return np.maximum(x,0)

# # [테스트 코드]
# for x in range(-3,4,1):
#     print(f"relu({x}) : {relu(x)}")

# 순전파를 수행하는 메서드 정의
def forward_neuralnet_hidden1(x):

    global pm_output, pm_hidden
    
    # hidden : 은닉계층의 각 노드가 갖는 값
    # 은닉계층의 가중치와 편향을 활용해 독립변수와의 연산 후 비선형 활성화 함수 relu() 적용
    hidden = relu(np.matmul(x, pm_hidden['w']) + pm_hidden['b'])

    # output : 출력계층의 각 노드가 갖는 값 
    # 출력 계층에 대한 선형 연산
    output = np.matmul(hidden, pm_output['w']) + pm_output['b']

    # 반환되어지는 값은 최종 연산 결괏값과 역전파에 활용되는 각 계층별 변수 
    return output, [x,hidden]

# # [테스트 코드]
# # 더미데이터 생성
# x = np.asarray([1.23, 0.31, 2.31])
# # 더미데이터에 따른 신경망 노구조 결정
# input_cnt = 3  
# set_hidden(2)
# output_cnt = 1

# # 파라미터 초기화 메서드 동작
# init_param()
# # 순전파 연산 메서드 동작 
# result = forward_neuralnet(x)

# print("순전파 연산 결괏값(output) : ", result[0])
# print("순전파 중간연산 결괏값(hidden) : ", result[1][1])
# print("독립변수(x) : ", result[1][0])

# 하나 이상의 은닉계층이 존재하는 경우 순전파 메서드 정의
# x : 독립변수
def forward_neuralnet_hiddens(x):
    # pm_hiddens : 입력계층과 은닉계층들 사이의 파라미터(리스트, 딕셔너리)
    # pm_output  : 마지막 은닉계층과 출력계층 사이의 파라미터(딕셔너리)
    global pm_output, pm_hiddens

    # 입력계층의 데이터(독립변수)
    hidden = x    # 연산용
    hiddens = [x] # 역전파 전달용 보조정보(기존값 포함)

    # 반복문을 통해 은닉계층들의 파라미터 정보를 하나씩 전달받도록 함 
    # 입력계층부터 마지막 은닉계층까지의 연산을 진행
    # 선형연산 결괏값 다음은 relu 적용 
    for pm_hidden in pm_hiddens:

        hidden = relu(np.matmul(hidden, pm_hidden['w']) + pm_hidden['b'])
        # 연산 결괏값의 할당
        hiddens.append(hidden)

    # 마지막 은닉계층까지의 연산결괏값과 출력계층의 파라미터의 연산
    # 활성화 함수 적용 X
    output = np.matmul(hidden, pm_output['w']) + pm_output['b']

    # 연산 결괏값(output) 반환 및 경사하강법 적용을 위해 hiddens 반환 
    return output, hiddens

# # 테스트 코드
# # 더미데이터 생성
# x = np.asarray([1.23, 0.31, 2.31, 0.3, 2.13])
# # 더미데이터에 따른 신경망 노구조 결정
# input_cnt = 5  
# set_hidden([3,2])
# output_cnt = 1

# # 파라미터 초기화 메서드 동작
# init_param()
# # 순전파 연산 메서드 동작 
# result = forward_neuralnet(x)

# print("독립변수(x) : ", result[1][0])
# print("첫 번째 은닉계층 연산 결괏값(hiddens) : ", result[1][1])
# print("두 번째 은닉계층 연산 결괏값(hiddens) : ", result[1][2])
# print("순전파 연산 결괏값(output) : ", result[0])

# 활성화 함수 ReLU 는 np.sign() 메서드로 구현할 수 있으며, 다음과 같은 특징을 갖습니다. 
# -1 if x < 0
# 0 if x==0
# 1 if x > 0
def relu_derv(y):
    return np.sign(y)

# 파라미터 갱신을 수행하는 메서드 
# G_output : 출력계층과 은닉계층 사이의 ∂L/∂Y
# aux : 입력 벡터, 은닉 계층의 정보 
def backprop_neuralnet_hidden1(G_output, aux):
    
    # pm_hidden : 입력계층과 은닉계층 사이의 파라미터
    # pm_output : 은닉계층과 출력계층 사이의 파라미터    
    global pm_output, pm_hidden
    
    # 입력 벡터, 은닉계층의 노드 값 각각 변수화 
    x, hidden = aux

    # [출력계층과 은닉계층 사이의 파라미터 갱신 준비과정]
    # 가중치 갱신 준비과정 : X^t * G
    g_output_w_output = hidden.transpose()
    G_w_out = np.matmul(g_output_w_output, G_output)
    # 편향 갱신 준비과정 : G
    G_b_out = np.sum(G_output, axis = 0)

    # [G_hidden 생성과정 1단계] 
    # G_Hidden : 은닉계층과 입력계층 사이의 ∂L/∂Y
    # 전체 수식 : (𝛅_k * w_k) * 𝜑(h)
    # 1단계에서 구현하는 수식 : (𝛅_k * w_k)
     
    # np.matmul() 연산 과정을 위해 행렬전환
    # G_hidden 을 구하기 위해서는 업데이트가 되지 않은 가중치가 필요하기에 
    # 가중치 업데이트 전에 활용
    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    # 출력계층과 은닉계층 사이의 파라미터 갱신
    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out


    # [G_hidden 생성과정 2단계] 
    # G_Hidden : 은닉계층과 입력계층 사이의 ₩
    # 전체 수식 : (𝛅_k * w_k) * 𝜑(h)
    # 2단계에서 구현하는 수식 : (𝛅_k * w_k) * 𝜑(h)
    # 은닉계층에서 비선형 활성화함수 relu가 사용되므로, 이에 따른 미분과정의 곱이 수행 
    G_hidden = G_hidden * relu_derv(hidden)

    # 은닉계층과 입력계층 사이의 파라미터 갱신준비과정
    # 가중치 갱신 준비과정
    g_hidden_w_hid = x.transpose()
    G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
    # 편향 갱신 준비과정
    G_b_hid = np.sum(G_hidden, axis=0)

    # 은닉계층과 입력계층 사이의 파리미터 갱신 
    pm_hidden['w'] -= LEARNING_RATE * G_w_hid
    pm_hidden['b'] -= LEARNING_RATE * G_b_hid

# 파라미터 업데이트 매서드 
# G_output : 가장 마지막 계층의 ∂L/∂Y (delta_k)
# aux : 독립변수, 은닉계층들이 갖는 노드 값 
def backprop_neuralnet_hiddens(G_output, aux):
    global pm_output, pm_hiddens

    # 독립변수와 은닉계층들의 노드값들을 hiddens 에 저장(리스트 타입)
    hiddens = aux 

    # [출력계층과 마지막 은닉계층 사이의 파라미터 갱신 준비]
    # 가중치 갱신에 필요한 수식 : ∂L/∂W (X^t * G)
    # 가장 마지막 은닉 계층(hiddens[-1])의 행렬전환 
    g_output_w_out = hiddens[-1].transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    # 편향 갱신에 필요한 수식 : ∂L/∂B (G)
    G_b_out = np.sum(G_output, axis = 0)

    # [마지막 은닉계층과 이전 은닉계층의 ∂L/∂Y (delta_k+1) 준비 - 1/2 단계]
    # 업데이트가 되지 않은 가중치가 필요 
    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    # [출력계층과 마지막 은닉계층 사이의 파라미터 갱신]
    # 출력계층과 마지막 은닉계층의 파라미터 업데이트
    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out

    # 마지막 은닉 계층과 그 이전 계층들 사이의 파라미터 업데이트 과정
    # 즉 뒤에서 부터 업데이트를 하기 위해 reversed() 를 활용
    # reversed() : 주어진 값들을 거꾸로 반환
    # pm_hiddens : 입력벡터 부터 마지막 은닉계층의 값을 담고 있음 (리스트)
    for n in reversed(range(len(pm_hiddens))):
        
        # [마지막 은닉계층과 이전 은닉계층의 ∂L/∂Y (delta_k+1) 준비 - 2/2단계]
        # 2단계에서 구현하는 수식 : (𝛅_k * w_k) * 𝜑(h)
        # 𝜑(h)에 매개변수는 가장 마지막 은닉계층의 값부터 순차적으로 들어와야함
        G_hidden = G_hidden * relu_derv(hiddens[n+1])

        # 가장 마지막 이전 계층의 파라미터 갱신 준비
        g_hidden_w_hid = hiddens[n].transpose()
        G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
        G_b_hid = np.sum(G_hidden, axis = 0)

        # 마지막 은닉계층의 이전 계층 ∂L/∂Y 연산 (𝛅_k * w_k)
        g_hidden_hidden = pm_hiddens[n]['w'].transpose()
        G_hidden = np.matmul(G_hidden, g_hidden_hidden)

        # 마지막 은닉계층 이전 계층의 파라미터 갱신
        pm_hiddens[n]['w'] -= LEARNING_RATE * G_w_hid
        pm_hiddens[n]['b'] -= LEARNING_RATE * G_b_hid

