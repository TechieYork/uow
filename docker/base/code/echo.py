# open file
resultFile = open("./data/result.txt", "wb")

# write result
resultFile.write(bytes("hello, docker pytorch-nvidia/n", 'utf-8'))

# close file
resultFile.close()
