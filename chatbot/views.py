from rest_framework.views import APIView
from rest_framework.response import Response
from .nlp_engine import get_answer

class AskView(APIView):
    def post(self, request):
        query = request.data.get("question", "")
        if not query:
            return Response({"error": "Please provide a question"}, status=400)
        
        answer = get_answer(query)
        return Response({"answer": answer})
