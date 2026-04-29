root = 'https://ngdp.sgs.gov.sa/NGDDS/DS_2000K/DS_2000K_GM_'
list_codes = ['F','G','H','I','J','K','L','M','N']
with open("input.txt", 'w') as f:
    for x in range(0,200):
        name1 = f'{root}{x:03d}.zip'
        name2 = f'{root}{x:03d}C.zip'
        print(name1, file=f)
        print(name2, file=f)
    for i in range(16,31):
        for j in list_codes:
            name3 = f'{root}{i}{j}.zip'
            print(name3, file=f)

