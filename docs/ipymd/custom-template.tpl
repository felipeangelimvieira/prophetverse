{%- extends 'markdown/index.md.j2' -%}

{%- block outputs -%}

<p class="cell-output-title jp-RenderedText jp-OutputArea-output">
{%- if cell.execution_count is defined -%}
    Output: <span class="cell-output-count">[{{ cell.execution_count|replace(None, "&nbsp;") }}]</span>
{%- else -%}
    Output: <span class="cell-output-count">[&nbsp;]</span>
{%- endif -%}
</p>
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