import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms.vertexai import VertexAI

if __name__=='__main__':
    print("hello LangChain")
    print(os.environ['OPENAI_API_KEY'])

    information="""
    Wardell Stephen Curry II (/ˈstɛfən/ STEF-ən; born March 14, 1988)[1] is an American professional basketball player for the Golden State Warriors of the National Basketball Association (NBA). Widely regarded as the greatest shooter and one of the greatest players of all time, Curry is credited with revolutionizing the sport by inspiring teams and players to take more three-point shots.[2][3][4][5] He is a four-time NBA champion, a two-time NBA Most Valuable Player (MVP), an NBA Finals MVP, an NBA All-Star Game MVP, and was named the inaugural NBA Western Conference Finals MVP. He is also a ten-time NBA All-Star, a nine-time All-NBA selection (including four on the First Team), and has won two gold medals at the FIBA World Cup as a member of the U.S. men's national team.
    Curry is the son of former NBA player Dell Curry and the older brother of current NBA player Seth Curry. He played collegiately for the Davidson Wildcats, where he set career scoring records for Davidson and the Southern Conference, and helped the Wildcats advance to the Elite Eight in 2008. He was named Conference Player of the Year twice, and set the NCAA single-season record for three-pointers made (162) during his sophomore year. Curry was selected by the Warriors as the seventh overall pick in the 2009 NBA draft.

    """
    
    #original prompt
    summary_template="""
    given information regarding a person, help me create :
    1. A summary
    2. Intresting facts about the person
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    #chatModel : which has llm model inside (wrapper)
    #llm=ChatOpenAI(temperature= 0, model_name='gpt-3.5-turbo') 
    llm=ChatOpenAI(temperature= 0, model_name='text-davinci-003') 
    #use model that is free

    #llm=VertexAI(model_name='text-bison@001')


    llm_chain=LLMChain(llm=llm, prompt=summary_prompt_template)
    result=llm_chain.invoke(input={"information": information})

    print(result)
