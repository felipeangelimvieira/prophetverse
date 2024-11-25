{%- extends 'markdown/index.md.j2' -%}

{%- block outputs -%}

<p class="cell-output-title">Output</p>
{{ super() }}


{%- endblock outputs -%}

{%- block stream -%}
    {%- if output.name == 'stdout' -%}
        {%- block stream_stdout -%}
        {%- endblock stream_stdout -%}
    {%- elif output.name == 'stdin' -%}
        {%- block stream_stdin -%}
        {%- endblock stream_stdin -%}
    {%- endif -%}
{%- endblock stream -%}