
# this is not a demo of LLMs but it is useful in our work

from textwrap import dedent

from RestrictedPython import safe_globals
from RestrictedPython import compile_restricted

xglob = 999

def demo_execute_arbitrary_code(mycompiler=compile):
    try:
        code = dedent("""\
        exec('a = 1 + 1', globals())
        """)
        byte_code = mycompiler(code, '<inline>', 'exec')
        exec(byte_code, globals())
        print("Exec arbitrary code: Success")
    except Exception as e:
        print(f"Exec arbitrary code: Failed ({e})")

def demo_file_system_access(mycompiler=compile):
    try:
        code = dedent("""
        with open('test.txt', 'w') as f:
            f.write('Testing file system access')
        """)
        byte_code = mycompiler(code, '<inline>', 'exec')
        exec(byte_code, globals())
        print("File system access: Success")
    except Exception as e:
        print(f"File system access: Failed ({e})")

def demo_network_access(mycompiler=compile):
    try:
        code = dedent("""\
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('example.com', 80))
        """)
        byte_code = mycompiler(code, '<inline>', 'exec')
        exec(byte_code, globals())
        print("Network access: Success")
    except Exception as e:
        print(f"Network access: Failed ({e})")

def demo_modify_environment(mycompiler=compile):
    try:
        code = dedent("""\
        import os
        os.environ['TEST'] = 'RestrictedPython'
        """)
        byte_code = compile_restricted(code, '<inline>', 'exec')
        exec(byte_code, globals())
        print("Modification of the environment: Success")
    except Exception as e:
        print(f"Modification of the environment: Failed ({e})")

def demo_builtin_eval(myglobals=globals()):  # note different globals
    try:
        eval('2 + 2', myglobals)  # just simple expr not involving globals()
        print("Built-in eval(): Success for 2+2")
    except Exception as e:
        print(f"Built-in eval(): Failed ({e}) for 2+2")
    try:
        eval('2 + xglob', myglobals)  # uses xglob from globals()
        print("Built-in eval(): Success for 2+xglob")
    except Exception as e:
        print(f"Built-in eval(): Failed ({e})  for 2+xglob")


if __name__ == '__main__':
    demo_execute_arbitrary_code()
    demo_execute_arbitrary_code(safe_globals)
    print("-"*50)
    demo_file_system_access()
    demo_file_system_access(compile_restricted)
    print("-"*50)
    demo_network_access()
    demo_network_access(compile_restricted)
    print("-"*50)
    demo_builtin_eval()
    demo_builtin_eval(safe_globals)  # note different globals
