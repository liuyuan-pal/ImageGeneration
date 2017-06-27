import tensorflow as tf

def get_var():
    with tf.variable_scope("test1"):
        var1 = tf.get_variable("test1_var1", [1])

        with tf.variable_scope("test2"):
            var2 = tf.get_variable("test2_var1", [1])

        var3 = tf.get_variable("test1_var2", [1])

        with tf.variable_scope("test2") as scope:
            scope.reuse_variables()
            var4 = tf.get_variable("test2_var1", [1])

    return var1,var2,var3,var4

if __name__=="__main__":
    with tf.variable_scope("test") as scope:
        var1,var2,var3,var4=get_var()
        print(var1.name,var2.name,var3.name,var4.name)
        scope.reuse_variables()
        var1, var2, var3,var4 = get_var()
        print(var1.name, var2.name, var3.name,var4.name)
