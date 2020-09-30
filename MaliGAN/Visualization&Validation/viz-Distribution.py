import matplotlib.pyplot as plt
sequence = []
generated_sequence = []
amp_file = open('AMP_Sequence.txt','r').read()
amp_generated = open('amp_generated.txt','r').read()
for _ in amp_file:
    if _ != '\n':
        sequence.append(_)
for _ in amp_generated:
    if _ != '\n':
        generated_sequence.append(_)
generated_sequence = [x.upper() for x in generated_sequence]

def count(file):
    P=M=F=G=C=R=L=Y=W=S=Q=N=A=E=T=D=H=K=I=V = 0
    for i in file:
        if i == 'P':
            P += 1
        elif i == 'M':
            M += 1
        elif i == 'F':
            F += 1
        elif i == 'G':
            G += 1
        elif i == 'C':
            C += 1
        elif i == 'R':
            R += 1
        elif i == 'L':
            L += 1
        elif i == 'Y':
            Y += 1
        elif i == 'W':
            W += 1
        elif i == 'S':
            S += 1
        elif i == 'Q':
            Q += 1
        elif i == 'N':
            N += 1
        elif i == 'A':
            A += 1
        elif i == 'E':
            E += 1
        elif i == 'T':
            T += 1
        elif i == 'D':
            D += 1
        elif i == 'H':
            H += 1
        elif i == 'K':
            K += 1
        elif i == 'I':
            I += 1
        elif i == 'V':
            V += 1
             #   else:
    return P,M,F,G,C,R,L,Y,W,S,Q,N,A,E,T,D,H,K,I,V

data_amp = []
data_generated = []
data_amp = count(sequence)
data_generated = count(generated_sequence)
x = ['P', 'M', 'F', 'G', 'C', 'R', 'L', 'Y', 'W', 'S', 'Q', 'N', 'A', 'E', 'T', 'D', 'H', 'K', 'I', 'V']
fig, axs = plt.subplots(2)
fig.suptitle('Distribution of AMP')
axs[0].bar(x, data_amp)
axs[0].set_title('Original')
axs[1].bar(x, data_generated)
axs[1].set_title('Generated')
fig.figsize=(500,500)
fig.savefig('Distribution.png')
plt.show()
