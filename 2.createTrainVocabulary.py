# -*- coding: utf-8 -*-
train_encode_file = 'train.enc'
train_decode_file = 'train.dec'
test_encode_file = 'test.enc'
test_decode_file = 'test.dec'

print('create start...')
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"
UNK = "__UNK__"
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

vocabulary_size = 5000
def gen_vocabulary_file(input_file, output_file):
	vocabulary = {}
	with open(input_file) as f:
		counter = 0
		for line in f:
			counter += 1
			tokens = [word for word in line.strip()]
			for word in tokens:
				if word in vocabulary:
					vocabulary[word] += 1
				else:
					vocabulary[word] = 1
		vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
		# 取前5000個字
		if len(vocabulary_list) > 5000:
			vocabulary_list = vocabulary_list[:5000]
		print(input_file + " 對話表size:", len(vocabulary_list))
		with open(output_file, "w") as ff:
			for word in vocabulary_list:
				ff.write(word + "\n")

gen_vocabulary_file(train_encode_file, "train_encode_vocabulary")
gen_vocabulary_file(train_decode_file, "train_decode_vocabulary")

train_encode_vocabulary_file = 'train_encode_vocabulary'
train_decode_vocabulary_file = 'train_decode_vocabulary'

print("對話轉向量...")
def convert_to_vector(input_file, vocabulary_file, output_file):
	tmp_vocab = []
	with open(vocabulary_file, "r") as f:
		tmp_vocab.extend(f.readlines())
	tmp_vocab = [line.strip() for line in tmp_vocab]
	vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])

	output_f = open(output_file, 'w')
	with open(input_file, 'r') as f:
		for line in f:
			line_vec = []
			for words in line.strip():
				line_vec.append(vocab.get(words, UNK_ID))
			output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
	output_f.close()

convert_to_vector(train_encode_file, train_encode_vocabulary_file, 'train_encode.vec')
convert_to_vector(train_decode_file, train_decode_vocabulary_file, 'train_decode.vec')

convert_to_vector(test_encode_file, train_encode_vocabulary_file, 'test_encode.vec')
convert_to_vector(test_decode_file, train_decode_vocabulary_file, 'test_decode.vec')
