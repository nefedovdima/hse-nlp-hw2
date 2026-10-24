"""Restricted AST diagnostic, not a Python sandbox or an official grader."""
import ast
import math
from e1.legacy_safe import SafeRunner, Unsupported, _BINARY


class NumericRunner(SafeRunner):
    def __init__(self):
        super().__init__(steps_limit=2000,length_limit=512)

    def _check(self,v):
        if type(v) in (int,float):
            if abs(v)>1e60 or not math.isfinite(v):
                raise Unsupported('numeric_bound')
            return v
        if type(v) is bool:
            return v
        if type(v) is str and len(v)<=512:
            return v
        if type(v) in (list,tuple) and len(v)<=512:
            for x in v:
                if type(x) not in (bool,int,float,str):
                    raise Unsupported('nested_container')
                self._check(x)
            return v
        raise Unsupported('value_type_or_size')

    def expr(self,node):
        if isinstance(node,ast.BinOp):
            self._tick()
            left,right = self.expr(node.left),self.expr(node.right)
            if type(left) not in (int,float,bool) or type(right) not in (int,float,bool):
                raise Unsupported('sequence_arithmetic')
            if type(node.op) not in _BINARY:
                raise Unsupported('operator')
            if isinstance(node.op,ast.Pow) and abs(right)>16:
                raise Unsupported('power_bound')
            return self._check(_BINARY[type(node.op)](left,right))
        return super().expr(node)

    def statements(self,body):
        # Route augmented arithmetic through exactly the same pre-allocation guards.
        rewritten=[]
        for s in body:
            if isinstance(s,ast.AugAssign):
                if not isinstance(s.target,ast.Name):
                    raise Unsupported('assignment_target')
                s=ast.Assign(targets=[s.target],value=ast.BinOp(
                    left=ast.Name(id=s.target.id,ctx=ast.Load()),op=s.op,right=s.value))
            rewritten.append(s)
        return super().statements(rewritten)

    def run(self,source):
        if len(source)>16000:
            raise Unsupported('source_length')
        tree=ast.parse(source)
        if sum(1 for _ in ast.walk(tree))>3000:
            raise Unsupported('ast_size')
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
            raise Unsupported('top_level_structure')
        f=tree.body[0]
        a=f.args
        if (f.name!='solve' or f.decorator_list or a.args or a.posonlyargs or a.kwonlyargs
                or a.vararg or a.kwarg or a.defaults or a.kw_defaults):
            raise Unsupported('signature')
        if f.returns is not None and not (isinstance(f.returns,ast.Name)
                and f.returns.id in ('int','float','str','bool','tuple','list')):
            raise Unsupported('annotation')
        done,value=self.statements(f.body)
        if not done:
            raise Unsupported('no_return')
        return self._check(value)


def run_numeric(source):
    try:
        value=NumericRunner().run(source)
        if type(value) not in (int,float):
            return 'nonnumeric',str(value)[:160]
        return 'ok',value
    except SyntaxError as e:
        return 'syntax_error',str(e)[:160]
    except (Unsupported,RecursionError) as e:
        return 'unsupported',str(e)[:160]
    except (ValueError,TypeError,ZeroDivisionError,OverflowError,IndexError,KeyError) as e:
        return 'runtime_error',type(e).__name__+': '+str(e)[:140]


def same_answer(pred,ref):
    return abs(pred-ref)<=1e-6+1e-6*abs(ref)
