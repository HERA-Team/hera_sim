{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in all_methods %}
      {%- if not (objname == 'Simulator' and item.startswith('add_')) and
            (not item.startswith('_') or item in ['__len__',
                                                  '__call__',
                                                  '__next__',
                                                  '__iter__',
                                                  '__getitem__',
                                                  '__setitem__',
                                                  '__delitem__',
                                                  ]) %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}

   {%- if objname=='Simulator' -%}
   .. rubric:: Deprecated Methods

   .. autosummary::
      :toctree: {{ objname }}
      {% for item in methods %}
         {%- if item.startswith('add_') %}
      ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {%- endif -%}

   {% endblock %}


   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
