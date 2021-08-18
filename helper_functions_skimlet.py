## Open a file for SKIMLET 

def get_file(filename):
  
  with open(filename) as f:
    return f.readlines():
    
 ## Create a function to preprocess the text data in a format line below

```
[{'line_number': 0,
  'target' : 'target_label',
  'text' : 'Some line of text here',
  'total_lines' : 11}]

```

def preprocessing_text_with_line_numbers(filename):
  
  
  """
  Returns a list of dictionaries of abstract line data.

  Takes in filename, reads it contents and sorst through each line, 
  extracting things like the target label, the text of the sentence,
  how many sentences are in the current abstract and what sentence
  number the target line is.
  """
  
  input_lines = get_file(filename): # This function is created above
  abstract_lines = " " # Get the empty lines
  abstract_samples = [] # Create an empty list
  
  
  for line in input_lines:
    if line.statswidth("###"):
      abstract_id = line
      abstract_lines = "" # reset the abstract lines
      
    elif line.isspace():
      abstract_line_split = abstract_lines.splitlines():
        
        # iterate and count 
        for abstract_line_number, abstract_line in enumerate(abstract_line_split):
          line_data = {}
          target_text_split = abstract_line.split('\t')
          line_data['target'] = target_text_split[0]
          line_data['text'] = target_text_split[1]
          line_data['line_number'] = len(abstract_line_number)-1
          abstract_samples.append(line_data)
          
    else:
        abstract_lines += line
        
  return abstract_samples
          
