from collections import defaultdict

lat = input()
assert (lat[-1] == 'n' or lat[-1] == 's')
dir1 = None
if lat[-1] == 'n':
    dir1 = "North"
else:
    dir1 = "South"
lat = lat[:-1]
longitude = input()
assert (longitude[-1] == 'e' or longitude[-1] == 'w')
dir2 = None
if longitude[-1] == 'e':
    dir2 = "East"
else:
    dir2 = "West"
longitude = longitude[:-1]

l = defaultdict(lambda: 0)

for i in lat:
    l[i] += 1

keymax1 = max(l, key=l.get)
keymin1 = min(l, key=l.get)

m = defaultdict(lambda: 0)

for j in longitude:
    m[j] += 1

keymax2 = max(m, key=m.get)
keymin2 = min(m, key=m.get)
ans1 = l[keymax1] - l[keymin1]
ans2 = m[keymax2] - m[keymin2]
print(ans1, dir1, ans2, dir2)




