import numpy as np

def entropy(value):
    return -value * np.log2(value)

def info_gain(entropy_parent, list_tuple):
    somme = 0
    for (prob, entro) in list_tuple:
        somme += prob * entro
    return entropy_parent - somme

a = entropy(5./12.)
b = entropy(3./12.)
c = entropy(4./12.)
maladie = a + b + c
print "entro maladie : ", maladie

n_un   = entropy(6./12.)
n_deux = entropy(6./12.)

print "info gain noyau : ", info_gain(maladie, [(6./12., n_un), (6./12., n_deux)])
print "entro noyau_1 : ", n_un
print "entro noyau_2 : ", n_deux

f_un   = entropy(7./12.)
f_deux = entropy(5./12.)
flagelle = f_un + f_deux
print "info gain flagelle : ", info_gain(maladie, [(7./12., f_un), (5./12., f_deux)])
print "entro flagelle_1 : ", f_un
print "entro flagelle_2 : ", f_deux

p = entropy(6./12.)
f = entropy(6./12.)
coloration = p + f
print "info gain coloration : ", info_gain(maladie, [(6./12., p), (6./12., f)])
print "entro coloration_p: ", p
print "entro coloration_f: ", f

fine = entropy(7./12.)
epai = entropy(5./12.)
paroi = fine + epai
print "info gain paroi : ", info_gain(maladie, [(7./12., fine), (5./12., epai)])
print "entro paroi_fine : ", fine
print "entro paroi_epai : ", epai
print "***********"

print "test gain : ", info_gain(0.940, [(7./14., .985), (7./14., .592)])

o_s = entropy(5./14.)
o_o = entropy(4./14.)
o_r = entropy(5./14.)
outlook = o_s + o_o + o_r
print "outlook : ", outlook

t_h = entropy(4./14.)
t_m = entropy(6./14.)
t_c = entropy(4./14.)
temperature = t_h + t_m + t_c
print "temperature : ", temperature

h_h = entropy(7./14.)
h_n = entropy(7./14.)
humidity = h_h + h_n
print "humidity : ", humidity

w_w = entropy(8./14.)
w_s = entropy(6./14.)
wind = w_w + w_s
print "wind : ", wind

pt_y = entropy(9./14.)
pt_n = entropy(5./14.)
play_tennis = pt_y + pt_n
print "play tennis : ", play_tennis
