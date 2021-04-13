import java.io.IOException;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.model.TextDataModel;
import net.librec.recommender.RecommenderContext;
import net.librec.similarity.CosineSimilarity;
import net.librec.similarity.RecommenderSimilarity;


public class LibRecDemo {

	public static void main(String[] args) throws LibrecException, InterruptedException, IOException {
		
		// Build data model 
		Configuration conf = new Configuration();
		
		// Utility configuration
		conf.set("rec.random.seed","1");						
		conf.set("rec.recommender.verbose","true");
		//conf.set("dfs.log.dir","");
		
		// Convertor
		conf.set("dfs.data.dir","data");						
        conf.set("data.input.path","filmtrust/ratings.txt");	
        conf.set("data.column.format","UIR");					
		conf.set("data.convert.binarize.threshold","-1.0");		
		conf.set("data.model.format","text");					
		
		// Splitter
		conf.set("data.model.splitter","ratio");			
		conf.set("data.splitter.ratio", "rating");
		conf.set("data.splitter.trainset.ratio", "0.8");
		
		DataModel dataModel = new TextDataModel(conf);
		dataModel.buildDataModel();
		
		// Build similarity
        conf.set("rec.recommender.similarity.key" ,"user");

        RecommenderSimilarity similarity = new CosineSimilarity();
        similarity.buildSimilarityMatrix(dataModel);

        // Build recommender context
        RecommenderContext context = new RecommenderContext(conf, dataModel, similarity);

        // Build recommender
        conf.set("rec.recommender.isranking" ,"false");
        
        conf.set("rec.iterator.learnrate","0.002");
        conf.set("rec.iterator.learnrate.maximum","0.05");
        conf.set("rec.iterator.maximum","100");
        conf.set("rec.user.regularization","0.01");
        conf.set("rec.item.regularization","0.01");
        conf.set("rec.impItem.regularization","0.01");
        conf.set("rec.bias.regularization","0.01");
        conf.set("rec.factor.number","20");
        conf.set("rec.learnrate.bolddriver","false");
        conf.set("rec.learnrate.decay","1.0");
                
        SVDPlusPlusRecommender recommender = new SVDPlusPlusRecommender();
        recommender.setContext(context);
        recommender.train(context);
                      
        int userId = 1;
        int itemId = 13;
        System.out.println("Predicted rating from user " + userId + " for item " + itemId + " is " + recommender.predict(userId, itemId));

	}

}
