import json
import copy
from pathlib import Path
import sys
import langdetect
langdetect.DetectorFactory.seed = 0


msgid_position = {}



def build_msgid_dict(msgs):
    d = {}
    for im, msg in enumerate(msgs):
        d[msg["message_id"]] = im
    return d


def find_msg_by_id(msgs, id):
    # global msgid_position
    
    return msgs[msgid_position[id]]
    
    for im, msg in enumerate(msgs):
        if msg["message_id"] == id:
            return msg
    
    return None


def get_node_text(node):
    return node["text"]


def extract_conversation(root, level, msg_list, msgs, full_conv_list, fully_translated, fully_validated):

    if root["message_id"] not in msgid_position:
        return full_conv_list, fully_translated, fully_validated 

    fmsg = find_msg_by_id(msgs, root["message_id"])
    text_key = "text"
        
    root_text = fmsg[text_key]

    # print(fmsg["labels"])
    # print(fmsg)
    if "quality" not in fmsg["labels"] or fmsg["labels"]["quality"]["value"] < 0.66:
        return full_conv_list, fully_translated, fully_validated

    if root_text not in msg_list:
        msg_list.append(root_text)
        if len(root["replies"]) == 0:
            if root["message_id"] not in msgid_position:
                return full_conv_list, fully_translated, fully_validated 

            crt = find_msg_by_id(msgs, root["message_id"])
            crt_list = []
            while crt.get("parent_id", None) != None:
                crt_list.append([crt["role"], get_node_text(crt)]) 
                crt = find_msg_by_id(msgs, crt["parent_id"])
           
            crt_list.append([crt["role"], get_node_text(crt)])
            full_conv_list.append(crt_list)
        for child in root["replies"]:
            extract_conversation(child, level+1, msg_list, msgs, full_conv_list, fully_translated, fully_validated)
        
    return full_conv_list, fully_translated, fully_validated
    
def get_lang_prob(text, lang="ro"):
    try:
        results = langdetect.detect_langs(text)
    except :
        return 1
    
    for res in results:
        if res.lang == lang:
            return res.prob
    return -1
  

def is_text_in_ro(text):
    in_ro = False
    prob = get_lang_prob(text)
    if prob > 0.65:
        in_ro = True
    elif (text.startswith("Cine") or text.startswith("Ce este") or text.startswith("Cum")) \
        and text.endswith("?"): 
        in_ro = True
    else:
        in_ro = False
    return in_ro
    # if prob != -1 and prob < 0.65:
    #     if in_ro == False:
    #         print(prob, in_ro, text[:75])


def pretty_print_conv(conv):
    print("-"*100)
    for c in conv[::-1]:
        print(c[1])
        print("-"*100)

if __name__ == "__main__":

    msg_file = Path("2023-04-12_oasst_all.messages_ro.jsonl")
    tree_file = Path("2023-04-12_oasst_all.trees_ro.json")

    with msg_file.open('r', encoding="utf-8") as f:
        msgs = f.readlines()
        msgs = list(map(lambda x: json.loads(x), msgs))
        ro_msgs = []
        for msg_index, msg in enumerate(msgs):
            # check language of message
            if msg_index % 20000 == 0:
                print("{0}/{1} - {2}".format(msg_index, len(msgs), len(ro_msgs)))
            if is_text_in_ro(msg["text"]):
                ro_msgs.append(msg)
            # if msg_index == 20000:
                # break
    
    msgs = ro_msgs
    msgid_position = build_msgid_dict(msgs)
    print("Total ro messages:", len(msgs))

    c = 0
    convs = []
    # train_convs, dev_convs, test_convs = [], [], []
    trees = json.load(tree_file.open('r', encoding="utf-8"))
    for tree in trees:
        if tree["tree_state"] != "ready_for_export":
            continue
        root = tree["prompt"]
        if root["message_id"] not in msgid_position:
            continue
        if len(root["replies"]) > 0:
            x, _, _ = extract_conversation(root, 0, [], msgs, [], [], [])
            convs.extend(x)
            if len(x) > 0:
                c += 1


    # sys.exit()
    import random
    for _ in range(2):
        pretty_print_conv(convs[random.randint(0, len(convs)-1)])
        print("###############")
    
    print("Total different convs in ro:", c)
    print("Total distinct threads in ro:", len(convs))

    

    json_convs = []
    for conv in convs:
        d = {}
        messages = []
        for c in conv[::-1]:
            messages.append({"content": c[1], "role": c[0]})
        if len(messages) % 2 == 1:
            messages = messages[:-1]
        # print(messages)
        d["messages"] = messages
        json_convs.append(d)
    
    print(len(json_convs))
    json.dump(json_convs, open("ro_sft_oasst.json", "w", encoding="utf-8"), indent=4)
        