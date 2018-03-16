class prod_ctype(object):
    def __init__(self,ipa,ipb):
        # Index of input A
        self.input_a = ipa
        # Index of input B
        self.input_b = ipb

def cmap(i,j,n):
    return (n * (n + 1) / 2) - ((n - i) * (n - i + 1) / 2) + (j - i)

def icmap(k,n):
    ii = 0
    for ii in range(n):
        if cmap(ii, n - 1, n) >= k :
            break

    j = k - cmap(ii, ii, n) + ii
    return prod_ctype(ii, j)




