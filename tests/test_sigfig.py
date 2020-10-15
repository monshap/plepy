from math import pi

from plepy.helper import sigfig

def test_sigfig__3sigfigs_of_pi_314():
    assert sigfig(pi, 3) == 3.14


def test_sigfig__3sigfigs_of_neg_pi_neg314():
    assert sigfig(-pi, 3) == -3.14
