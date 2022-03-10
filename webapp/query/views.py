from django.shortcuts import render
import e15190

from e15190.runlog.query import Query as e15190_Query

def home(request):
    run_input = request.GET.get('input-run', '4082')
    run = int(run_input)
    run_info = dict()
    for key, value in e15190_Query.get_run_info(run).items():
        run_info[key] = str(value)
    context = {
        'run_info': run_info,
    }
    return render(request, 'query/home.html', context)
