def run_dynamic_code(user_input):\n    eval(user_input)\n    exec(f'print({user_input})')
