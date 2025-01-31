import os
import wandb
from datasets import load_dataset
from langchain.prompts import PromptTemplate

def generate_template(num_choices):
	"""
	Generate a template for multiple choice questions based on the number of choices.

	Args:
		num_choices (int): The number of answer choices (e.g., 2, 3, 4).

	Returns:
		str: A formatted template string.
	"""
	choices = "\n".join([f"{chr(65 + i)}) {{{chr(97 + i)}}}" for i in range(num_choices)])
	template = f"""Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [{', '.join([chr(65 + i) for i in range(num_choices)])}].\n\n{{prompt}}\n{choices}\n\nAnswer:"""
	return template

template1 = "Complete the sentence with the most plausible ending. Answer should be one among [A, B, C, D].\n\n{prompt}\nA) {a}\nB) {b}\nC) {c}\nD {d}\n\n\nAnswer:"

template2 = "Answer the following question by selecting the most appropriate option. Answer should be one among [A, B].\n\n{prompt}\nA) {a}\nB) {b}\n\n\nAnswer:"

prompt4_1 = PromptTemplate(
	template=template1, 
	input_variables=['prompt','a','b','c','d']
)

prompt2_1 = PromptTemplate(
	template=template2, 
	input_variables=['prompt','a','b']
)

prompt2 = PromptTemplate(
	template=generate_template(2), 
	input_variables=['prompt','a','b']
)
prompt3 = PromptTemplate(
	template=generate_template(3), 
	input_variables=['prompt','a','b','c']
)
prompt4 = PromptTemplate(
	template=generate_template(4), 
	input_variables=['prompt','a','b','c','d']
)

def format_text(example, data_name):
	'''
	Format an prompt into text
	'''
	if data_name == 'boolq':
		text = f"Passage: {example['passage']}\nQuestion: {example['question']}"
		example['gt'] = chr(ord('A') + int(example['answer']))
		example['num_choices'] = 2
		return {
			"text": prompt2.format(
				prompt=text,
				a="No",
				b="Yes"
			)
		}
	
	elif data_name == 'piqa':
		text = f"Question: {example['goal']}"
		example['gt'] = chr(ord('A') + example['label'])
		example['num_choices'] = 2
		return {
			"text": prompt2.format(
				prompt=text,
				a=example['sol1'],
				b=example['sol2']
			)
		}
	
	elif data_name == 'social_i_qa':
		text = f"Context: {example['context']}\nQuestion: {example['question']}"
		example['gt'] = chr(ord('A') + int(example['label']) - 1)
		example['num_choices'] = 3
		return {
			"text": prompt3.format(
				prompt=text,
				a=example['answerA'],
				b=example['answerB'],
				c=example['answerC']
			)
		}
	
	elif data_name == 'openbookqa':
		choices = example['choices']['text']
		text = f"Question: {example['question_stem']}"
		example['gt'] = example['answerKey']
		example['num_choices'] = 4
		return {
			"text": prompt4.format(
				prompt=text,
				a=choices[0],
				b=choices[1],
				c=choices[2],
				d=choices[3]
			)
		}
		
	elif data_name == 'allenai/ai2_arc':
		choices = example['choices']['text']
		if len(example['choices']['label']) != 4:
			choices.append('')
		text = f"Question: {example['question']}"
		example['gt'] = example['answerKey']
		example['num_choices'] = 4
		return {
			"text": prompt4.format(
				prompt=text,
				a=choices[0],
				b=choices[1],
				c=choices[2],
				d=choices[3]
			)
		}
	
	elif data_name == 'hellaswag':
		choices = example['endings']
		text = f"Context: {example['ctx']}"
		example['gt'] = chr(ord('A') + int(example['label']))
		example['num_choices'] = 4
		return {
			"text": prompt4_1.format(
				prompt=text,
				a=choices[0],
				b=choices[1],
				c=choices[2],
				d=choices[3]
			)
		}
		
	elif data_name == 'winogrande':
		text = f"Context: {example['sentence']}"
		example['gt'] = chr(ord('A') + int(example['answer']) - 1)
		example['num_choices'] = 2
		return {
			"text": prompt2_1.format(
				prompt=text,
				a=example['option1'],
				b=example['option2']
			)
		}
		
	else:
		raise NotImplementedError
		
		
def initialize_eval_table(dataset_name):
	
	if dataset_name == "boolq":
		columns = ['Question', 'Passage', 'Ground Truth', 'Prediction', 'Inference Time (ms)', 'Text']
	elif dataset_name == "piqa":
		columns = ['Goal', 'Solution A', 'Solution B', 'Ground Truth', 'Prediction', 'Inference Time (ms)', 'Text']
	elif dataset_name == "social_i_qa":
		columns = ['Context', 'Question', 'Answer A', 'Answer B', 'Answer C', 'Ground Truth', 'Prediction', 'Inference Time (ms)', 'Text']
	elif dataset_name in "openbookqa":
		columns = ['Question_Stem', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Ground Truth', 'Prediction', 'Inference Time (ms)', 'Text']
	elif dataset_name in "allenai/ai2_arc":
		columns = ['Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Ground Truth', 'Prediction', 'Inference Time (ms)', 'Text']
	elif dataset_name == "hellaswag":
		columns = ['Context', 'Ending A', 'Ending B', 'Ending C', 'Ending D', 'Ground Truth', 'Prediction', 'Inference Time (ms)', 'Text']
	elif dataset_name == "winogrande":
		columns = ['Sentence', 'Option A', 'Option B', 'Ground Truth', 'Prediction', 'Inference Time (ms)', 'Text']
	else:
		raise ValueError(f"Dataset {dataset_name} is not supported.")
		
	return wandb.Table(columns=columns)


def add_to_eval_table(eval_table, data, dataset_name, prediction, infer_time):

	if dataset_name == "boolq":
		eval_table.add_data(
			data['question'],
			data['passage'],
			data['gt'],  # Ground truth
			prediction,  # Predicted answer
			infer_time,
			data['text']
		)
	elif dataset_name == "piqa":
		eval_table.add_data(
			data['goal'],
			data['sol1'],
			data['sol2'],
			data['gt'],  # Ground truth
			prediction,
			infer_time,
			data['text']
		)
	elif dataset_name == "social_i_qa":
		eval_table.add_data(
			data['context'],
			data['question'],
			data['answerA'],
			data['answerB'],
			data['answerC'],
			data['gt'],  # Ground truth
			prediction,
			infer_time,
			data['text']
		)
	elif dataset_name in "openbookqa":
		eval_table.add_data(
			data['question_stem'],
			data['choices']['text'][0],
			data['choices']['text'][1],
			data['choices']['text'][2],
			data['choices']['text'][3],
			data['gt'],  # Ground truth
			prediction,
			infer_time,
			data['text']
		)
	elif dataset_name in "allenai/ai2_arc":
		eval_table.add_data(
			data['question'],
			data['choices']['text'][0],
			data['choices']['text'][1],
			data['choices']['text'][2],
			data['choices']['text'][3],
			data['gt'],  # Ground truth
			prediction,
			infer_time,
			data['text']
		)
	elif dataset_name == "hellaswag":
		eval_table.add_data(
			data['ctx'],
			data['endings'][0],
			data['endings'][1],
			data['endings'][2],
			data['endings'][3],
			data['gt'],  # Ground truth
			prediction,
			infer_time,
			data['text']
		)
	elif dataset_name == "winogrande":
		eval_table.add_data(
			data['sentence'],
			data['option1'],
			data['option2'],
			data['gt'],  # Ground truth
			prediction,
			infer_time,
			data['text']
		)
	else:
		raise ValueError(f"Dataset {dataset_name} is not supported.")
		