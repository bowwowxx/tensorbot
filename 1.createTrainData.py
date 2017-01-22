import os
import random

conv_path = 'dgk_shooter_min.conv'

if not os.path.exists(conv_path):
	print('dgk_shooter_min.conv error')
	exit()

convs = []
with open(conv_path, encoding = "utf8") as f:
	one_conv = []
	for line in f:
		line = line.strip('\n').replace('/', '')
		if line == '':
			continue
		if line[0] == 'E':
			if one_conv:
				convs.append(one_conv)
			one_conv = []
		elif line[0] == 'M':
			one_conv.append(line.split(' ')[1])

ask = []
response = []
for conv in convs:
	if len(conv) == 1:
		continue
	if len(conv) % 2 != 0:
		conv = conv[:-1]
	for i in range(len(conv)):
		if i % 2 == 0:
			ask.append(conv[i])
		else:
			response.append(conv[i])

print(len(ask), len(response))
print(ask[:3])
print(response[:3])


def convert_seq2seq_files(questions, answers, TESTSET_SIZE = 8000):

    train_enc = open('train.enc','w')
    train_dec = open('train.dec','w')
    test_enc  = open('test.enc', 'w')
    test_dec  = open('test.dec', 'w')


    test_index = random.sample([i for i in range(len(questions))],TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i % 1000 == 0:
            print(len(range(len(questions))), 'process:', i)

    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()

convert_seq2seq_files(ask, response)
