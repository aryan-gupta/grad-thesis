

auto_1 = {
    "0" : ["0", "1", "3"],
    "1" : ["1", "3"],
    "2" : ["1", "2", "3"],
    "3" : ["1", "2", "3"]
}

auto_2 = {
    "0" : ["0", "1"],
    "1" : ["2"],
    "2" : ["1", "2"]
}


auto_final = {}
key_f = []

for key_1 in auto_1.keys():
    for key_2 in auto_2.keys():
        key_f.append((key_1, key_2))

print(key_f)

for src_1, src_2 in key_f:
    for dest_1, dest_2 in key_f:
        if dest_1 in auto_1[src_1] and dest_2 in auto_2[src_2]:
            key_f_src_str = src_1 + src_2
            key_f_dest_str = dest_1 + dest_2
            if key_f_src_str not in auto_final.keys(): auto_final[key_f_src_str] = []
            auto_final[key_f_src_str].append(key_f_dest_str)

for key in auto_final:
    print(key, end=" : ")
    print(auto_final[key])
