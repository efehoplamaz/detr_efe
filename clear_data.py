import os
import json
from collections import Counter
import shutil


TR_ANN_PATH = '/home/s1764306/data/annotations/BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json'
TST_ANN_PATH = '/home/s1764306/data/annotations/BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json'

TR_AUDIO_PATH = '/home/s1764306/data/audio/mc_2018/audio/'
TST_AUDIO_PATH = '/home/s1764306/data/audio/mc_2019/audio/'


with open(TR_ANN_PATH) as json_file:
    train_data = json.load(json_file)

with open(TST_ANN_PATH) as json_file2:
    test_data = json.load(json_file2)

#####################################################################################

train_missing_anns = os.listdir(TR_AUDIO_PATH)
test_missing_anns = os.listdir(TST_AUDIO_PATH)

print('Initially the length of training annotations is {} and length of testing annotations is {}.'.format(len(train_data), len(test_data)))

print('There are {} many training audio files'.format(len(train_missing_anns)))
print('There are {} many testing audio files'.format(len(test_missing_anns)))

d_species_train = {}
d_species_test = {}

for i, d in enumerate(train_data):

	if d['issues'] == True:
		print('Issue found!')
		continue

	if d['id'] in train_missing_anns and d['annotation']:
		train_missing_anns.remove(d['id'])

	if d['annotation']:
		if d['class_name'] in d_species_train:
			d_species_train[d['class_name']] += 1
		else:
			d_species_train[d['class_name']] = 1

for i, d in enumerate(test_data):

	if d['issues'] == True:
		print('Issue found!')
		continue

	if d['id'] in test_missing_anns and d['annotation']:
		test_missing_anns.remove(d['id'])

	if d['annotation']:
		if d['class_name'] in d_species_test:
			d_species_test[d['class_name']] += 1
		else:
			d_species_test[d['class_name']] = 1

tr_spicies = ([e[0] for e in d_species_train.items()])
tst_spicies = ([e[0] for e in d_species_test.items()])

diff = list(set(tst_spicies) - set(tr_spicies))

diff_sp_counts = {s: int(d_species_test[s]*0.8) for s in diff}

print()

print('There are {} training audio files which does not have any annotation!'.format(len(train_missing_anns)))
print('There are {} testing audio files which does not have any annotation!'.format(len(test_missing_anns)))
print('Species which are in test data but not in training are {}'.format(diff))
print('The species will be moved from test to train are {}'.format(diff_sp_counts))

print()


################################## DECIDING WHICH FILES TO MOVE FROM TEST TO TRAIN

total_switched = {s: 0 for s in diff}

tst_to_train = []

for i, d in enumerate(test_data):

	if (d['class_name'] in diff) and (total_switched[d['class_name']] < diff_sp_counts[d['class_name']]) and (d['annotation']):
		total_switched[d['class_name']] += 1
		tst_to_train.append(d)
		os.rename(TST_AUDIO_PATH + d['id'], TR_AUDIO_PATH + d['id'])

test_data = [d for d in test_data if not any([elm['id'] == d['id'] for elm in tst_to_train])]
train_data = train_data + tst_to_train

print('After moving files from test to train there are {} many training annotations, {} many testing annotations.'.format(len(train_data), len(test_data)))
print('After moving files from test to train there are {} many training audio files, {} many testing audio files.'.format(len(os.listdir(TR_AUDIO_PATH)), len(os.listdir(TST_AUDIO_PATH))))
print()

###################################################################################

#print(int(len(os.listdir(TR_AUDIO_PATH))*0.2), len(train_missing_anns))

tst_rm_empty = test_missing_anns[30:]

test_data = [d for d in test_data if not any([elm == d['id'] for elm in tst_rm_empty])]

for elm in tst_rm_empty:
	os.rename(TST_AUDIO_PATH + elm, TR_AUDIO_PATH + elm)

print('After moving empty files from test to train there are {} many training annotations, {} many testing annotations.'.format(len(train_data), len(test_data)))
print('After moving empty files from test to train there are {} many training audio files, {} many testing audio files.'.format(len(os.listdir(TR_AUDIO_PATH)), len(os.listdir(TST_AUDIO_PATH))))
print()

####################################################################################

tr_sp = {}
tst_sp = {}

tr_audio_files = os.listdir(TR_AUDIO_PATH)
tst_audio_files = os.listdir(TST_AUDIO_PATH)

b_in_train_audio = []
b_in_test_audio = []

tr_no_aud_file = []
tst_no_aud_file = []

for d in train_data:

	if d['id'] not in tr_audio_files:
		tr_no_aud_file.append(d['id'])

	if d['id'] in tr_audio_files and d['annotation']:
		tr_audio_files.remove(d['id'])

	if d['annotation']:
		if d['class_name'] in tr_sp:
			tr_sp[d['class_name']] += 1
		else:
			tr_sp[d['class_name']] = 1

for d in test_data:

	if d['id'] not in tst_audio_files:
		tst_no_aud_file.append(d['id'])

	if d['id'] in tst_audio_files and d['annotation']:
		tst_audio_files.remove(d['id'])

	if d['annotation']:
		if d['class_name'] in tst_sp:
			tst_sp[d['class_name']] += 1
		else:
			tst_sp[d['class_name']] = 1

if tr_no_aud_file:
	print('Removing train annotations which does not have any audio file.')
	train_data = [d for d in train_data if d['id'] not in tr_no_aud_file]
if tst_no_aud_file:
	print('Removing test annotations which does not have any audio file.')
	test_data = [d for d in test_data if d['id'] not in tst_no_aud_file]


print("There are {} many training audio files, {} many audio files without annotation, {} many annotations.".format(len(os.listdir(TR_AUDIO_PATH)), len(tr_audio_files), len(train_data)))
print("There are {} many testing audio files, {} many audio files without annotation, {} many annotations.".format(len(os.listdir(TST_AUDIO_PATH)), len(tst_audio_files), len(test_data)))

print()

print('All spicies in test are also in training: {}'.format((list(set(tst_sp) - set(tr_sp)) == [])))
print('All annotations in training JSON has a matching audio file: {}'.format(all([True if d['id'] in os.listdir(TR_AUDIO_PATH) else False for d in train_data])))
print('All annotations in testing JSON has a matching audio file: {}'.format(all([True if d['id'] in os.listdir(TST_AUDIO_PATH) else False for d in test_data])))


with open('/home/s1764306/data/annotations/train.json', 'w') as outfile:
    json.dump(train_data, outfile)

with open('/home/s1764306/data/annotations/test.json', 'w') as outfile2:
    json.dump(test_data, outfile2)






























#print(all([(elm in os.listdir(TR_AUDIO_PATH) and elm not in os.listdir(TST_AUDIO_PATH))for elm in tst_to_train]))
# # remove_list_test = test_missing_anns[30:]
# add_list_training = os.listdir('./audio/empty/wavs/')[:(120-len(train_missing_anns))]


# TRAIN_AUDIO = 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2018/audio/'
# TST_AUDIO = 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2019/audio/'
# EMPTY_AUDIO = 'C:/Users/ehopl/Desktop/bat_data/audio/empty/wavs/'

# for elm in remove_list_test:
# 	os.remove(TST_AUDIO + elm)

# for elm in add_list_training:
# 	shutil.copy(EMPTY_AUDIO + elm, TRAIN_AUDIO + elm)


# ########################################################################

# train_missing_anns = os.listdir('./audio/mc_2018/audio/')
# test_missing_anns = os.listdir('./audio/mc_2019/audio/')

# print('After moving the files from empty to train, there are {} many training audio files'.format(len(train_missing_anns)))
# print('After removing some audio files from test, there are {} many testing audio files'.format(len(test_missing_anns)))


