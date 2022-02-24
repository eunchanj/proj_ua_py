#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def p_cal(
    box1,
    box2,
    reject_mean=True
):
    """
    Compute p-value for hypothesis that given statistic of score function for indicators1 is higher than for
    indicators2 using bootstrapping.
    box1 : indicators1 by Bootstrap, example : auc, 10000 times bootstrap -> 10000 aucs
    box2 : indicators2 by Bootstrap
    reject_mean : when indicator2 > indiciators1
    """
    p_sub = np.subtract(box1, box2)
    p_mean = np.subtract(box1, box2).mean()
    if reject_mean and p_mean<0:
        print("mean < 0, swap box1 and box2 ")
    else:
        p_sub_mean = np.subtract(p_sub, p_mean)
        p = ((percentileofscore(p_sub_mean, -p_mean, kind="weak")/100.0) + (1-percentileofscore(p_sub_mean, p_mean, kind="weak")/100.0))
        return p

