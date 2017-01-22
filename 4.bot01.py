# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.models.rnn.translate import seq2seq_model
import os
import numpy as np
import time
from slackclient import SlackClient

BOT_ID = os.environ["BOT_ID"]
AT_BOT = "<@" + BOT_ID + ">"
EXAMPLE_COMMAND = "orange"
slack_token = os.environ["SLACK_TOKEN"]
slack_client = SlackClient(slack_token)

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encode_vocabulary = 'train_encode_vocabulary'
train_decode_vocabulary = 'train_decode_vocabulary'

def handle_command(command, channel):
    response = str(command)
    if command.startswith(EXAMPLE_COMMAND):
       response = "Sure...write some more code then I can do that!"
    slack_client.api_call("chat.postMessage", channel = channel, text = response, as_user = True)

def parse_slack_output(slack_rtm_output):
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
               return output['text'].split(AT_BOT)[1].strip().lower(),output['channel']
    return None, None

def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
         tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab

vocab_en, _, = read_vocabulary(train_encode_vocabulary)
_, vocab_de, = read_vocabulary(train_decode_vocabulary)

vocabulary_encode_size = 5000
vocabulary_decode_size = 5000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256
num_layers = 3
batch_size = 1

model = seq2seq_model.Seq2SeqModel(source_vocab_size = vocabulary_encode_size, target_vocab_size = vocabulary_decode_size,
    buckets = buckets, size = layer_size, num_layers = num_layers, max_gradient_norm = 5.0,
    batch_size = batch_size, learning_rate = 0.5, learning_rate_decay_factor = 0.99, forward_only = True)
model.batch_size = 1
READ_WEBSOCKET_DELAY = 1

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('.')
if ckpt != None:
    print(ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
else :
    print("none ckpt modle")

if slack_client.rtm_connect():
    print("StarterBot connected and running!")
    while True:
        command, channel = parse_slack_output(slack_client.rtm_read())
        if command and channel:
            print type(command)
            input_string = command.encode("utf8")
            input_string_vec = []
            for words in input_string.strip():
                 input_string_vec.append(vocab_en.get(words, UNK_ID))
            bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vec)])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis = 1)) for logit in output_logits]
            if EOS_ID in outputs:
                outputs = outputs[: outputs.index(EOS_ID)]

            response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs])
            print("response:" + response)
            # handle_command(response, channel) unicode(unicodestring,"utf-8") response.encode("utf8")
            slack_client.api_call("chat.postMessage", channel = channel, text = unicode(response,"utf-8"), as_user = True)
        time.sleep(READ_WEBSOCKET_DELAY)

else :
     print("Connection failed. Invalid Slack token or bot ID?")
