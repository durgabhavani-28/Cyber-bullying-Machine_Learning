from django import forms

class InputTextForm(forms.Form):
    input_text = forms.CharField(widget=forms.Textarea(attrs={'rows': 5, 'cols': 30}))
