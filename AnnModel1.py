import csv 
import numpy as np
from mlp import *

# 정규분포를 갖는 난수값을 생성하기 위한 매개변수 정의 
RND_MEAN, RND_STD = 0.0, 1.0
LEARNING_RATE = 0.01

# 메서드 정의
def main(epoch_count:int=10, mb_size:int=10, report:int=1, train_ratio:float=0.8):

    # 함수 호출관계에 맞춰 하위 메서드들을 배치합니다. 
    load_dataset()
    init_param()

	# 본 메서드의 매개변수는 AI 모델 학습을 위한 하이퍼파라미터이며, 
    # 이를 수행하는 rain_and_test() 메서드의 매개변수로 할당합니다. 
    train_and_test(epoch_count, mb_size, report, train_ratio)


# 메서드 정의
def load_dataset(): 

    # 본 메서드의 동작 순서
    # 1)파일 열기  2)파일 읽기  3-1)파일 전처리 및 변수화  3-2)실험용 Data 생성
    
    # 1)파일 열기
    # open() 메서드를 통해 csv 파일을 열고 해당 변수를 csvfile 이라 지정
    with open('./Regression_data.csv') as csvfile: 
        # 2)파일 읽기
        # csvfile 변수를 csv.reader() 메서드를 통해 읽고 변수화
        csvreader = csv.reader(csvfile) 
       
        # 3-1)파일 전처리 및 변수화
        # 본 파일의 경우 첫 번째 행은 변수 이름 이므로 이를 건너뛰어 줍니다. 
        next(csvreader) 
       
        # append() 메서드를 통해 값을 저장하기 위한 변수
        rows = [] 
        # 반복문 및 append() 을 통해 값을 하나씩 저장
        for row in csvreader:
            rows.append(row)
    
    # 3-2) 실험용 Data 생성 
    # 추후 다른 메서드에서 편하게 사용하기 위한 전역 변수 선언
    # 데이터, 입력 및 출력 계층의 크기
    global data, input_cnt, output_cnt 

    # 입력 및 출력 계층의 크기 지정
    input_cnt, output_cnt = 10, 1
    
    # 임시로 데이터를 저장하기 위한 버퍼를 생성 합니다.
    # 버퍼의 행은 데이터의 개수, 열은 입출력 계층의 수(10+1)
    data = np.zeros([len(rows), input_cnt + output_cnt])

    # 반복문 및 enumerate() 메서드를 통해 버퍼에 본 데이터를 채워넣습니다. 
    # 문자열 정보('M', 'F', 'I')를 숫자정보(100, 010, 001)로 치환합니다.
    for n, row in enumerate(rows):
        
        # 만약 n 번째 행의 첫 번째 열이 문자 'M' 이라면 버퍼의 [0,0] 위치에 1의 숫자정보를 할당. [1,0,0,...]
        # 만약 n 번째 행의 첫 번째 열이 문자 'F' 이라면 버퍼의 [0,1] 위치에 1의 숫자정보를 할당. [0,1,0,...]
        # 만약 n 번째 행의 첫 번째 열이 문자 'I' 이라면 버퍼의 [0,2] 위치에 1의 숫자정보를 할당. [0,0,1,...]
        if row[0] == 'M': 
            data[n, 0] = 1

        if row[0] == 'F':
            data[n, 1] = 1

        if row[0] == 'I': 
            data[n, 2] = 1

        # 이후 나머지 열의 숫자정보(row[1:]) 는 [n, 3:] 위치로 버퍼를 채워줍니다.
        data[n, 3:] = row[1:]

    # 메서드 정의 
def init_param():
    
    # 추후 다른 메서드에서 편하게 활용하기 위해 전역변수로 정의 
    global weight, bias

    # 가중치 행렬 값들을 np.random.normal() 함수를 이용해 정규분포를 갖는 난수값으로 초기화 합니다. 
    # 행과 열은 [10,1] 로 맞춰 주며, 평균값 0.0, 표준편차 1.0의 정규분포를 따르도록 합니다. 
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])

    # 편향의 경우 하나의 값을 갖는 스칼라로 np.zeros() 메서드를 통해 0의 값을 갖도록 합니다.
    # 편향은 초기에 지나친 영향을 주어 학습에 역효과를 불러오지 않도록 0으로로 초기화
    bias = np.zeros([output_cnt])

# 메서드 정의 
def arrange_data(mb_size, train_ratio):
    
    # 활용 빈도가 높은 변수들은 전역변수로 할당하며, 활용빈도가 낮은 변수의 경우 반환처리 하였습니다. 
    global data, shuffle_map, test_begin_index

    '''세부 기능 1)전체 데이터의 인덱싱 생성 후 셔플링.'''
    # 실험에 쓰일 데이터(data)의 수에 맞게 인덱싱(0,1,2,3,...)을 생성하는 과정 입니다.
    # 입출력 예시
    # >>>data.shape[0] 
    # 20
    shuffle_map = np.arange(data.shape[0])
    # 학습의 효율을 높여주기 위해 인덱싱(shuffle_map)을 뒤섞어 줍니다.
    np.random.shuffle(shuffle_map)

    '''세부 기능 2)1에폭에 필요한 미니배치의 수 연산.'''
    # 1 에폭을 위한 미니배치의 수(mini_batch_step_count)는 
    # 전체 데이터(data.shape[0])에서 학습 데이터의 비율(train_ratio)만큼의 개수를 구한 후 
    # 미니배치 크기(mb_size)로 나눠 몫 만을 구해줍니다. 
    # ( ※ mb_size 는 하나의 미니배치에 포함된 데이터의 수 입니다.)
    mini_batch_step_count = int(data.shape[0] * train_ratio) // mb_size

    '''세부 기능 3)학습 데이터와 테스트 데이터의 경계 인덱스를 구함.'''
    # 전체 데이터에서 학습 데이터와 테스트 데이터가 나뉘는 경계가 되는 인덱스를 
    # 테스트 데이터가 시작되는 인덱스로 설정하였습니다. 
    # 이 값을 구하기 위해서는 다양한 방식이 있지만, 
    # 미니배치 스탭 수(mini_batch_step_count)와 미니배치 크기(mb_size)를 곱해 
    # 학습 및 테스트 데이터 경계 인덱스(test_begin_index)를 구해줍니다.
    test_begin_index = mini_batch_step_count * mb_size

    # 다수의 변수가 생성되었지만 mini_batch_step_count 변수만 밖으로 반환합니다.
    # 활용 빈도가 높은 변수들은 전역변수로 할당하며, 활용빈도가 낮은 변수의 경우 반환처리 하였습니다. 
    return mini_batch_step_count

# 메서드 정의
def get_train_data(mb_size, nth):
    
    # 학습 데이터의 경우 새로운 Epoch 의 학습이 수행되기 전(if nth == 0)
    # 학습 데이터 인덱싱 정보를 무작위로 섞어(np.random.shuffle()) 학습의 효과를 높여줍니다.
    # 학습의 효과를 높여주는 선행 연구에 근거함
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_index])

    # 학습 데이터의 분리는 미니배치를 고려해야 합니다. 
    # 전체 데이터(data)에서 뒤섞인 인덱싱값으로 접근(shuffle_map)하여 미니배치 처리를 수행합니다.
    # 만약 미니배치 사이즈(mb_size) 값이 4 라면 다음과 같이 나눠볼 수 있습니다. 
    # 0 : 4
    # 4 : 8
    # 8 : 12
    # 12 : 16
    # 그리고 이와 같은 배수의 연산을 위한 수식을 미니배치 크기(mb_size)와 몇 번째 미니배치 처리 단계(nth) 값을 활용합니다.
    train_data = data[shuffle_map[mb_size * nth : mb_size * (nth+1)]]
    
    # 미니배치에 따라 나눠진 학습 데이터를 입출력 벡터로 나눠 반환합니다. 
    return train_data[:,:-output_cnt], train_data[:,-output_cnt:]

# 메서드 정의
def get_test_data():

    """ 전체 데이터에서 테스트 데이터 분리 과정 """
    # 전체 데이터(data)의 인덱싱 정보를 갖고 있는 변수(shuffle_map)에 접근하여 테스트 데이터를 얻어냅니다. 
    # 테스트 데이터의 경우 테스트 데이터의 시작 인덱싱 위치(test_begin_index)를 기준으로 인덱싱 끝 까지 범위를 갖도록 합니다. 
    test_data = data[shuffle_map[test_begin_index:]]

    """ 테스트 데이터에서 입출력 벡터 분리 과정 """
    # test_data 변수는 입출력 벡터를 포함한 테스트 데이터 입니다. 
    # 그렇기에 입출력 벡터를 분리하기 위한 과정을 수행하여야 합니다. 
    # 출력 계층 값을 담는 변수(output_cnt)를 활용하여 슬라이싱 과정을 수행합니다. 
    # 행은 모두 포함하지만 열의 경우 다음과 같이 구분할 수 있습니다.  
    return test_data[:,:-output_cnt], test_data[:,-output_cnt:]

# 메서드 정의 
def train_and_test(epoch_count, mb_size, report, train_ratio):
    
    '''1) 데이터 셔플링 - arrange_data()'''
    # 메서드를 통해 데이터를 섞어주고 1 에폭에 필요한 미니배치 수 를 반환합니다. 
    mini_batch_step_count = arrange_data(mb_size,train_ratio)

    '''2) 테스트 데이터 분리 - get_test_data()'''
    # 학습 데이터는 미니배치를 고려하여야 하기에 
    # 미니배치 고려가 필요 없는 테스트 데이터 분리 메서드를 먼저 배치합니다. 
    test_x, test_y = get_test_data()

    # 다음은 학습 및 테스트 수행 단계를 정의하는 과정 입니다.
    # 학습은 사용자가 지정한 학습횟수(에폭-epoch_count)에 맞춰 학습을 반복하도록 정의합니다. 
    for epoch in range(epoch_count):

        # 학습 진행에 따른 정보를 담기 위해 빈 리스트를 사전에 정의하여 줍니다. 
        losses, accs = [], []

        # 1 에폭이 수행되기 위해서는 미니배치 처리에 따른 학습이 모두 완료되어야 합니다. 
        # 그리고 에폭은 사용자가 1 이상의 값을 설정할 수 있어야 하므로, 
        # 이를 위한 이중 반복문 구조를 설계할 수 있습니다. 
        # 만약 학습 데이터의 수가 16개, 미니배치 크기(mb_size)가 4개라면 
        # 1 에폭을 위해서는 총 4번(mini_batch_step_count)의 미니배치에 따른 학습이 수행되어야 합니다. 
        for nth in range(mini_batch_step_count):
            
            '''3) 학습 데이터 분리 - get_train_data()'''
            # 학습 데이터의 경우 에폭마다의 미니배치 처리 단계(nth)를 고려하여 학습 데이터의 입출력 벡터를 변수화 합니다. 
            train_x, train_y = get_train_data(mb_size, nth)
            
            '''4) 학습 - run_train()'''
            # 미니배치 처리 단계(nth)에 따른 학습 데이터의 입출력 벡터를 학습합니다.
            # 지금은 run_train() 메서드의 이름만 정의해둔 상황입니다. 
            # 정리하면 미니배치 단계에 따라 임의로 설정한 고정된 실험 결과값을 반환합니다. 
            #TODO run_train() 메서드는 다음 과정에서 하위 메서드 정의
            loss, acc = run_train(train_x,train_y)
            
            # 미니배치 단계에 따른 실험 결괏값(loss, acc)들을 append() 메서드로 묶어줍니다. 
            losses.append(loss)
            accs.append(acc)

        '''5) 테스트 - run_test()'''
        # 테스트 과정은 사용자가 지정한 1 이상의 결과 보고 주기(report)에 따른 에폭 마다 수행하도록 합니다. 
        # 모든 에폭마다 테스트를 수행할 수도 있지만, 그건 효율적인 방식이 되지 못합니다. 
        
        # 상황 예제 1) 
        # 만약 학습 반복 주기(에폭)를 10 이라 하며, 결과 보고 주기(report)를 2 라 하면 
        # 다음과 같은 주기로 테스트 결과를 반환하도록 합니다.
        # 0(x), 1(o), 2(x), 3(o), 4(x), 5(o), 6(x), 7(o), 8(x), 9(o) 
        
        # 상황 예제 2) 
        # 만약 학습 반복 주기(에폭)를 10 이라 하며, 결과 보고 주기(report)를 3 이라 하면 
        # 다음과 같은 주기로 테스트 결과를 반환하도록 합니다.
        # 0(x), 1(x), 2(o), 3(x), 4(x), 5(o), 6(x), 7(x), 8(o), 9(x) 
        
        # 이와같은 조건을 갖는 주기를 반환할 수 있도록 코드를 작성합니다. 

        # 이러한 경우 나머지를 반환하는 연산자와 비교 연산자를 활용할 수 있습니다. 
        if report > 0 and (epoch+1) % report == 0:
            
            # 위 조건에 부합하는 경우 테스트를 진행합니다. 
            acc = run_test(test_x, test_y)

            # 수행한 결과를 사용자에게 출력합니다. 
            # 현재 에폭
            # 현재 에폭에 따른 미니배치 단계별 결과의 학습 손실(loss) 평균값
            # 현재 에폭에 따른 미니배치 단계별 결과의 학습 정확도(accs) 평균값
            # 현재 에폭의 정확도(loss)
            print("Epoch {}   : Train - Loss = {:.3f}, Accuracy = {:.3f} / Test - Accuracy = {:.3f}".\
                  format(epoch+1, np.mean(losses), np.mean(accs), acc))
            
        
    '''5) 최종 테스트 - run_test()'''
    # 학습을 모두 마쳤다면 주어진 조건에 다른 모델 파라미터에 대한 조정이 완료되었다는 의미 입니다. 
    # 그렇기에 최종 테스트를 수행하여 학습을 모두 마친 AI 모델에 대한 성능 평가를 run_test() 메서드로 수행합니다. 
    final_acc = run_test(test_x, test_y)

    # 학습에 따른 최종 결괏값 출력
    print("="*30, ' Final TEST ', '='*30)
    print('\nFinal Accuracy = {:.3f}'.format(final_acc))

# 매서드 정의 
def forward_neuralnet(x):
		
    y_hat = np.matmul(x, weight) + bias
		
    return y_hat, x

# 매서드 정의 
def forward_postproc(y_hat, y):
    diff = y_hat - y
    square = np.square(diff)
    loss = np.mean(square)
	   
    return loss, diff

# 매서드 정의 
def eval_accuracy(y_hat, y):
    
	# 오차율 구하는 과정    
	# np.mean() 메서드의 이유는 미니배치 처리를 고려하여 하나의 지표로 묶어주기 위함 입니다. 
    mdiff = np.mean(np.abs((y_hat - y) / y))
    # 1 에서 오차율을 빼 정확도를 구합니다. 
    return 1 - mdiff

# backprop_neuralnet() 메서드 2차 정의 (편향 경사하강법)
def backprop_neuralnet(G_output, x):
    # 갱신되는 값은 가중치 뿐 아니라 편향도 함께 이뤄집니다. 
	# 그렇기에 이 두 값을 모두 전역변수로 지정하여 다음에 파라미터를 사용할 때는 
	# 업데이트 된 파라미터를 사용할 수 있도록 합니다. 
	
    global weight, bias
		
	#가중치의 경사하강법 수식 구현
	# 입력 벡터의 전치행렬이 필요하므로 transpose() 메서드를 활용합니다. 
    x_transpose = x.transpose()

	# 전치된 입력 벡터(x_transpose)와 손실함수(L)를 순전파의 결괏값(Y)에 대하여 편미분한 결과(G_output)의 행렬곱셈을 수행합니다. 
    G_w = np.matmul(x_transpose, G_output)
		

    '''편향의 경사하강법 수식 구현'''
	# 편향의 정리된 수식에 맞춰 행렬로 이뤄진 G_output 값을 np.sum()을 통해 연산처리 합니다. 
    G_b = np.sum(G_output, axis = 0)
		
	# 기존 가중치와 편향값을 업데이트하기 위한 대입 연산자를 활용하여 경사하강법 수식을 코드화 합니다. 
    weight -= LEARNING_RATE * G_w
    bias   -= LEARNING_RATE * G_b

# backprop_postproc() 메서드 3차 정의(3/3)
def backprop_postproc(diff):
	
	# 앞서 M과 N의 값은 편차의 행렬정보를 바탕으로 구할 수 있었습니다. 
	# 편차의 정보(diff)를 매개변수로 전달받아 이를 변수처리 하겠습니다. 
	# 추후 diff 를 얻는 메서드에 대하여 diff 를 반환처리 할 수 있도록 수정하겠습니다.
    M_N = diff.shape
		
    # Square에 관한 MSE의 편미분 결과에 맞춰 다음과 같이 코드를 작성합니다. 
    # 1 값을 반환하는 np.ones() 메서드와 주어진 축에 대한 배열 요소의 곱을 반환하는 np.prod() 메서드를 활용하였습니다. 
    g_mse_square = np.ones(M_N) / np.prod(M_N)
    # Diff에 관한 Square의 편미분 결과에 맞춰 다음과 같이 코드를 작성합니다. 
    g_square_diff = 2 * diff
    # Output에 관하여 Diff의 편미분 결과에 맞춰 다음과 같이 코드를 작성합니다.
    g_diff_output = 1
    
    # 편미분을 통해 구해준 두 결괏값에 대하여 곱해줍니다.
    G_diff = g_mse_square * g_square_diff
    # 편미분을 통해 구해준 결괏값들을 순차적으로 곱해줍니다. 
    G_output = g_diff_output * G_diff

    # 미분을 모두 마친 결괏값을 경사하강법에 활용하고자 값을 반환하여 줍니다.
    return G_output

# 메서드 재정의 
def run_train(x, y):
    # 1) 가장 먼저 순전파 연산이 수행되어야 합니다. 그렇기에 해당 메서드를 배치합니다. 
    # auxiliary(보조하다) 의 약자로 변수명에 aux 를 활용합니다. 
    # neuralnet의 약자 nn 을 활용합니다. 
    y_hat, aux_nn_x = forward_neuralnet(x)
		
    # 2) 순전파의 결괏값(예측값)을 받아 AI 모델의 손실값을 확인합니다. 
    # postproc의 약자 pp 를 활용합니다. 
    loss, aux_pp_diff = forward_postproc(y_hat, y)
    
    # 3) 예측값과 실제값을 활용하여 AI 모델의 정확도를 확인합니다. 
    accuracy = eval_accuracy(y_hat, y)

    # 4) 순전파에 따른 여러 중간 지표를 확인하였으니 역전파를 수행하여 파라미터를 수정합니다. 
    # 다만 역전파에 앞서 손실함수의 미분 G_output(G) 를 구하여 줍니다. 
    G_output = backprop_postproc(aux_pp_diff)

    # 5) G_output을 구하였으니, 실제 파라미터가 업데이트 되는 메서드를 배치합니다. 
    # 해당 메서드는 위의 다른 메서드와 달리 반환되어지는 값이 없습니다. 
    # 파라미터의 업데이트 후 전역변수를 취하였기에, 다음 순전파에서는 업데이트된 파라미터를 활용할 수 있습니다. 
    backprop_neuralnet(G_output, aux_nn_x)
    
    # 위에서 구한 손실값과 정확도는 반환시켜 사용자에게 보여줄 수 있도록 취합니다. 
    return loss, accuracy

# 매서드 재정의 
def run_test(x, y):
    # 1) 순전파의 결괏값인 예측값을 수행하는 메서드를 배치합니다. 
    # 여기서 언더바 '_' 의 의미는 해당 메서드에서 불필요하게 사용되어지는 변수기에
    # 언더바 처리를 통해 숨겨주었습니다. (※ 협업 관계자와의 약속이 필요)
    y_hat, _ = forward_neuralnet(x)
		
    # 2) 순전파의 결괏값과 실제 결괏값을 활용하여 정확도를 구해주는 메서드를 배치합니다. 
    accuracy = eval_accuracy(y_hat, y)

    # 정확도에 대하여 반환 처리를 수행합니다.  
    return accuracy

if __name__ == "__main__":
    set_hidden([1,2])
    main()