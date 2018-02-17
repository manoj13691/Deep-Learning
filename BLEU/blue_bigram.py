def blue(mt, references):
	# It takes "mt" as a string which is the machine translated output.
	# "References" represent the set of output which is provided by humans
	# that represents the probable translations 

	from nltk import bigrams
	mt_bigrams_count = {}
	references_max_bigrams_count = {}

	mt = mt.lower()
	for i in range(len(references)):
		references[i] = references[i].lower()


	mt_bigram_tuple_list = list(bigrams(mt.split()))
	for i in mt_bigram_tuple_list:
		if i in mt_bigrams_count:
			mt_bigrams_count[i] += 1
		else:
			mt_bigrams_count[i] = 1



	#Initialize references_bigrams_count to zero
	for i in mt_bigrams_count:
		if i not in references_max_bigrams_count:
			references_max_bigrams_count[i] = 0

	reference_bigram_count_list = []
	for i in references:
		bi = list(bigrams(i.split()))
		dict = {}
		for j in bi:
			if j in dict:
				dict[j] += 1
			else:
				dict[j] = 1
		reference_bigram_count_list.append(dict) 


	for i in references_max_bigrams_count:
		max_count = 0
		for j in reference_bigram_count_list:
			if i in j:
				max_count = max( max_count, j[i])
		references_max_bigrams_count[i] = max_count

	
	sum_count = 0
	sum_count_clip = 0

	for i in mt_bigrams_count:
		sum_count += mt_bigrams_count[i]


	for i in references_max_bigrams_count:
		sum_count_clip += references_max_bigrams_count[i]

	return round(float(sum_count_clip)/float(sum_count),2)
	

print blue("The cat the cat on the mat", ["The cat is on the mat", "There is a cat on the mat"])