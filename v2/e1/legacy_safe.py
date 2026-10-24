"""Evaluation-only restricted arithmetic interpreter for HW2 validation programs.

No exec/eval/subprocess/imports of generated code; never use outputs as training labels.
Unsupported programs are marked separately (not graded wrong).
"""
import ast
import math
import operator


class Unsupported(Exception):
    pass


_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


class SafeRunner:
    def __init__(self, steps_limit=2000, length_limit=5000):
        self.steps_limit = steps_limit
        self.length_limit = length_limit
        self.steps = 0
        self.env = {}

    def _tick(self):
        self.steps += 1
        if self.steps > self.steps_limit:
            raise Unsupported('steps_limit')

    def _check(self, v):
        if isinstance(v, bool):
            return v
        if type(v) in (int, float):
            if abs(v) > 1e100 or (isinstance(v, float) and not math.isfinite(v)):
                raise Unsupported('value_limit')
            return v
        if type(v) is str and len(v) <= self.length_limit:
            return v
        if type(v) in (list, tuple) and len(v) <= self.length_limit:
            for x in v:
                if type(x) not in (bool, int, float, str, tuple, list):
                    raise Unsupported('unsupported_item')
            return v
        raise Unsupported('unsupported_value')

    def expr(self, node):
        self._tick()
        if isinstance(node, ast.Constant):
            return self._check(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise Unsupported('unknown_variable')
            return self.env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            vals = [self.expr(n) for n in node.elts]
            return self._check(tuple(vals) if isinstance(node, ast.Tuple) else vals)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left = self.expr(node.left)
            right = self.expr(node.right)
            if isinstance(node.op, ast.Pow) and (not isinstance(right, (int, float)) or abs(right) > 20):
                raise Unsupported('unsafe_power')
            if type(left) is str or type(right) is str:
                if not isinstance(node.op, ast.Add):
                    raise Unsupported('unsafe_string_operation')
            return self._check(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return self._check(_UNARY[type(node.op)](self.expr(node.operand)))
        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            for op, n in zip(node.ops, node.comparators):
                if type(op) not in _COMPARE:
                    raise Unsupported('comparison')
                right = self.expr(n)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            v = self.expr(node.values[0])
            for n in node.values[1:]:
                if (isinstance(node.op, ast.And) and not v) or (isinstance(node.op, ast.Or) and v):
                    break
                v = self.expr(n)
            return self._check(v)
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test) else node.orelse)
        if isinstance(node, ast.Subscript):
            seq = self.expr(node.value)
            if type(seq) not in (list, tuple, str):
                raise Unsupported('subscript_nonsequence')
            ix = self.expr(node.slice)
            if type(ix) is not int:
                raise Unsupported('index_noninteger')
            return self._check(seq[ix])
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise Unsupported('function_call')
            name = node.func.id
            args = [self.expr(a) for a in node.args]
            if name == 'range':
                if not (1 <= len(args) <= 3) or any(type(a) is not int for a in args):
                    raise Unsupported('range_args')
                r = range(*args)
                if len(r) > self.length_limit:
                    raise Unsupported('range_limit')
                return list(r)
            funcs = {'abs': abs, 'min': min, 'max': max, 'round': round,
                     'sum': sum, 'len': len, 'int': int, 'float': float}
            if name not in funcs or len(args) > 4:
                raise Unsupported('call_' + name)
            return self._check(funcs[name](*args))
        raise Unsupported('expression_' + type(node).__name__)

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            if len(target.id) > 200:
                raise Unsupported('varname_length')
            self.env[target.id] = self._check(value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)) or len(target.elts) != len(value):
                raise Unsupported('unpack_length')
            for t, v in zip(target.elts, value):
                self.assign(t, v)
            return
        raise Unsupported('assignment_target')

    def statements(self, body):
        for stmt in body:
            self._tick()
            if isinstance(stmt, ast.Return):
                return True, self.expr(stmt.value) if stmt.value is not None else None
            if isinstance(stmt, ast.Assign):
                v = self.expr(stmt.value)
                for t in stmt.targets:
                    self.assign(t, v)
            elif isinstance(stmt, ast.AnnAssign):
                if stmt.value is None:
                    raise Unsupported('annotation_without_value')
                self.assign(stmt.target, self.expr(stmt.value))
            elif isinstance(stmt, ast.AugAssign):
                if not isinstance(stmt.target, ast.Name) or type(stmt.op) not in _BINARY:
                    raise Unsupported('aug_assign')
                name = stmt.target.id
                if name not in self.env:
                    raise Unsupported('unknown_variable')
                value = _BINARY[type(stmt.op)](self.env[name], self.expr(stmt.value))
                self.assign(stmt.target, value)
            elif isinstance(stmt, ast.If):
                done, value = self.statements(stmt.body if self.expr(stmt.test) else stmt.orelse)
                if done:
                    return True, value
            elif isinstance(stmt, ast.For):
                items = self.expr(stmt.iter)
                if type(items) not in (tuple, list):
                    raise Unsupported('for_iterable')
                for item in items:
                    self.assign(stmt.target, item)
                    done, value = self.statements(stmt.body)
                    if done:
                        return True, value
                done, value = self.statements(stmt.orelse)
                if done:
                    return True, value
            elif isinstance(stmt, ast.While):
                for _ in range(self.steps_limit):
                    if not self.expr(stmt.test):
                        done, value = self.statements(stmt.orelse)
                        if done:
                            return True, value
                        break
                    done, value = self.statements(stmt.body)
                    if done:
                        return True, value
                else:
                    raise Unsupported('loop_limit')
            elif isinstance(stmt, (ast.Pass, ast.Expr)):
                if isinstance(stmt, ast.Expr) and not isinstance(stmt.value, ast.Constant):
                    self.expr(stmt.value)
            else:
                raise Unsupported('statement_' + type(stmt).__name__)
        return False, None

    def run(self, source):
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise Unsupported('top_level_structure')
        func = tree.body[0]
        if func.name != 'solve' or func.args.args or func.args.kwonlyargs:
            raise Unsupported('solve_signature')
        done, answer = self.statements(func.body)
        if not done:
            raise Unsupported('no_return')
        return self._check(answer)


def run_safe(source):
    try:
        return 'ok', SafeRunner().run(source)
    except (SyntaxError, Unsupported, ZeroDivisionError, OverflowError,
            ValueError, TypeError, IndexError, KeyError) as e:
        return 'error', type(e).__name__ + ': ' + str(e)[:80]


def same_answer(a, b, tol=1e-6):
    if type(a) in (int, float, bool) and type(b) in (int, float, bool):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if type(a) in (list, tuple) and type(b) in (list, tuple):
        return len(a) == len(b) and all(same_answer(x, y, tol) for x, y in zip(a, b))
    return a == b
