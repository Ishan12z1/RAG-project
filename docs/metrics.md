Evaluation : 

1. Retrieval
    1. Recall@5 : It tells us if there is atleast one relvant document out of the top 5 retrieved documents.   
        formula : retrieved_docs intersection with ground truth , if this is null then 0 else 1.   
        Recall@5= (1 / N) * Σ hit(q) over all N queries  
        Interpretation: Recall@5 = 0.80 means 80% of questions had at least one relevant chunk in the top 5.
    2. MRR (Mean Reciprocal Rank) : what is the mean rank of the first relevant query from the top 5 queries. 
        rank(q) = position (1,2,3,...) of the first relevant retrieved chunk  
        If none found: score = 0  
        RR(q) = 1 / rank(q) (reciprocal rank)  
        MRR = Sum(RR(q))/N  
        first relevant usually at rank 1 → MRR near 1.0  
        usually rank 2 → ~0.5  
        usually rank 5 → ~0.2  
2. Groundedness
    1. Supported = every factual claim in the answer is backed by the retrieved evidence (and if you use citations, the cited chunk actually contains that fact).  
    2. Unsupported = if even one factual claim is not backed (missing evidence, wrong citation, invented detail), the whole answer is labeled Unsupported.  
    3. Abstained = the model clearly says it cannot answer from the evidence and does not guess.  
    so supported is good, unsupported is bad and abstained is condition good when the query truley can't be answered from the relevant context else bad.   
    Steps for calculating :   
        1. Split the answer and take the lines that has some claims.  
        2. check if all the claims are genuin (supported and by a relevant document)  
        3. if all claims are genuin the groundedness will be 1 for this output.      
           if a single calim is not genuin then the output will be marked as 0.   
    
    formula = 1 / N * Σ unsupported(q)      
    7/9 means 7 claims are not supported  


