

auto_1 = ["a", "b", "c", "d"]
auto_2 = ["0", "1", "2"]

def auto_1_valid(src, dest, ops):
    if src == "a": return True if dest == "b" or dest == "c" else False
    if src == "b": return True if dest == "a" or dest == "d" else False
    if src == "c": return True if dest == "a" or dest == "d" else False
    if src == "d": return True if dest == "b" or dest == "c" else False

def auto_2_valid(src, dest, ops):
    if src == "0" and dest =="0": return True if "c" not in ops else False
    if src == "0" and dest =="1": return True if "b" in ops and "c" in ops else False
    if src == "0" and dest =="2": return True if "b" in ops and "c" not in ops else False
    if src == "1" and dest =="0": return False
    if src == "1" and dest =="1": return True
    if src == "1" and dest =="2": return False
    if src == "2" and dest =="0": return False
    if src == "2" and dest =="1": return True if "c" in ops else False
    if src == "2" and dest =="2": return True if "c" not in ops else False

auto_final = {}
key_f = []

for key_1 in auto_1:
    for key_2 in auto_2:
        key_f.append((key_1, key_2))

print(key_f)

for src_1, src_2 in key_f:
    for dest_1, dest_2 in key_f:
        if auto_1_valid(src_1, dest_1, []) and auto_2_valid(src_2, dest_2, [dest_1]):
            key_f_src_str = src_1 + src_2
            key_f_dest_str = dest_1 + dest_2
            if key_f_src_str not in auto_final.keys(): auto_final[key_f_src_str] = []
            auto_final[key_f_src_str].append(key_f_dest_str)

for key in auto_final:
    print(key, end=" : ")
    print(auto_final[key])
