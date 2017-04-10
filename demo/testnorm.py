# import random
# mu1 = 0.0
# sigma1 = 1.0
# mu2 = 0.0
# sigma2 = 1.0
# iter_cnt = 10**7
# in_cnt = 0
# for i in range(iter_cnt):
#     x = random.normalvariate(mu1, sigma1)
#     y = random.normalvariate(mu2, sigma2)
#     if (x*x + y*y + -1) ** 2 - (x*y)**2 < 0:
#         in_cnt += 1
#
# print(float(in_cnt)/iter_cnt)


def g(x1, x2):
    w0 = 3.5
    w1 = 5.6
    w2 = 2.5

    return w0+w1*x1+w2*x2

if __name__ == "__main__":
    print(g(-3.0, 2.1))
    print(g(1.5, 3.5))
    print(g(-2, 3.08))