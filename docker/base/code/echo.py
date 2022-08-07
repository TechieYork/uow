# open file
resultFile = open("./data/result.txt", "wb")

# write result
resultFile.write(bytes("hello, docker pytorch/n", 'utf-8'))

# close file
resultFile.close()
