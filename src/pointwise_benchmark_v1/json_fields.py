"""Lexically skip unrequested top-level JSON values without decoding them.

This is intentionally a projection reader, not a full JSON validator. Input
identity is checked separately. Object keys are decoded; only explicitly named
top-level values are passed to json.loads. Opaque values are scanned for string
escapes and container boundaries without interpreting numbers or string content.
"""
import json
from pathlib import Path


def string_end(text,start):
    i=start+1;escaped=False
    while i<len(text):
        char=text[i]
        if escaped:escaped=False
        elif char=='\\':escaped=True
        elif char=='"':return i+1
        i+=1
    raise ValueError('unterminated JSON string')


def value_end(text,start):
    if text[start]=='"':return string_end(text,start)
    if text[start] not in '[{':
        i=start
        while i<len(text) and text[i] not in ',}':i+=1
        return i
    stack=[];i=start
    while i<len(text):
        char=text[i]
        if char=='"':i=string_end(text,i);continue
        if char in '[{':stack.append(char)
        elif char in ']}':
            if not stack or (stack.pop(),char) not in (('[',']'),('{','}')):raise ValueError('unbalanced JSON container')
            if not stack:return i+1
        i+=1
    raise ValueError('unterminated JSON container')


def top_fields(path,wanted):
    text=Path(path).read_text();wanted=set(wanted);output={};i=0
    def space(index):
        while index<len(text) and text[index].isspace():index+=1
        return index
    i=space(i)
    if text[i]!='{':raise ValueError('top-level JSON object required')
    i=space(i+1)
    while i<len(text) and text[i]!='}':
        if text[i]!='"':raise ValueError('quoted top-level key required')
        end=string_end(text,i);key=json.loads(text[i:end]);i=space(end)
        if text[i]!=':':raise ValueError('JSON colon required')
        start=space(i+1);end=value_end(text,start)
        if key in wanted:
            if key in output:raise ValueError('duplicate requested top-level key')
            output[key]=json.loads(text[start:end])
        i=space(end)
        if text[i]==',':i=space(i+1)
        elif text[i]!='}':raise ValueError('JSON value separator required')
    if i>=len(text) or text[i]!='}' or space(i+1)!=len(text):raise ValueError('invalid top-level JSON ending')
    if set(output)!=wanted:raise ValueError('missing requested top-level fields')
    return output
